"""
Phase 1: logits -> next token.

logits
   |
repetition penalty
   |
repeat-ngram block
   |
temperature
   |
top-k
   |
top-p
   |
softmax
   |
sample (or argmax if greedy)
   |
next_token
"""

import torch


class Sampler:
    def __init__(
        self,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        repeat_ngram_size: int = 0,
        greedy: bool = False,
    ):
        if temperature <= 0:
            raise ValueError("temperature must be > 0")

        if top_k < 0:
            raise ValueError("top_k must be >= 0")

        if not (0.0 < top_p <= 1.0):
            raise ValueError("top_p must be in (0, 1]")

        if repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be > 0")

        if repeat_ngram_size < 0:
            raise ValueError("repeat_ngram_size must be >= 0")

        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.repeat_ngram_size = repeat_ngram_size
        self.greedy = greedy

    def sample(
        self,
        logits: torch.Tensor,
        generated_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        logits:
            [batch, vocab]

        generated_tokens:
            [batch, sequence] or None

        returns:
            next_token [batch]
        """

        if logits.dim() != 2:
            raise ValueError(
                f"logits must be [batch, vocab], got shape {tuple(logits.shape)}"
            )

        logits = logits.clone()

        batch_size = logits.shape[0]

        if generated_tokens is None:
            generated_tokens = torch.zeros(
                (batch_size, 0),
                dtype=torch.long,
                device=logits.device,
            )

        logits = self._apply_repetition_penalty(logits, generated_tokens)
        logits = self._apply_repeat_ngram_block(logits, generated_tokens)

        if self.greedy:
            return torch.argmax(logits, dim=-1)

        logits = logits / self.temperature
        logits = self._apply_top_k(logits)
        logits = self._apply_top_p(logits)

        probs = torch.softmax(logits, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)

        return next_token.squeeze(-1)

    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        generated_tokens: torch.Tensor,
    ) -> torch.Tensor:

        if self.repetition_penalty == 1.0 or generated_tokens.shape[1] == 0:
            return logits

        for b in range(logits.shape[0]):
            seen = torch.unique(generated_tokens[b])

            token_logits = logits[b, seen]

            penalized = torch.where(
                token_logits > 0,
                token_logits / self.repetition_penalty,
                token_logits * self.repetition_penalty,
            )

            logits[b, seen] = penalized

        return logits

    def _apply_repeat_ngram_block(
        self,
        logits: torch.Tensor,
        generated_tokens: torch.Tensor,
    ) -> torch.Tensor:

        n = self.repeat_ngram_size

        if n <= 0:
            return logits

        seq_len = generated_tokens.shape[1]

        if seq_len < n - 1:
            return logits

        for b in range(logits.shape[0]):
            tokens = generated_tokens[b].tolist()

            if len(tokens) < n - 1:
                continue

            prefix_size = n - 1
            current_prefix = tuple(tokens[-prefix_size:]) if prefix_size > 0 else ()

            banned_tokens = set()

            for i in range(len(tokens) - prefix_size):
                window = tuple(tokens[i:i + prefix_size])

                if window == current_prefix:
                    banned_tokens.add(tokens[i + prefix_size])

            for token_id in banned_tokens:
                logits[b, token_id] = float("-inf")

        return logits

    def _apply_top_k(self, logits: torch.Tensor) -> torch.Tensor:

        if self.top_k <= 0 or self.top_k >= logits.shape[-1]:
            return logits

        top_k_values, _ = torch.topk(logits, self.top_k, dim=-1)

        threshold = top_k_values[:, -1].unsqueeze(-1)

        logits = torch.where(
            logits < threshold,
            torch.full_like(logits, float("-inf")),
            logits,
        )

        return logits

    def _apply_top_p(self, logits: torch.Tensor) -> torch.Tensor:

        if self.top_p >= 1.0:
            return logits

        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)

        sorted_probs = torch.softmax(sorted_logits, dim=-1)

        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Keep the smallest set of tokens whose cumulative probability
        # exceeds top_p. Shift by one position so the first token that
        # crosses the threshold is still kept.
        sorted_indices_to_remove = cumulative_probs > self.top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False

        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1,
            index=sorted_indices,
            src=sorted_indices_to_remove,
        )

        logits = logits.masked_fill(indices_to_remove, float("-inf"))

        return logits
