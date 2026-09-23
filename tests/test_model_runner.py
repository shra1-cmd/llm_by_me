import torch

from src.inference.model_runner import ModelRunner
from src.inference.sampler import Sampler


class FakeConfig:
    def __init__(self, max_seq_len=None):
        self.max_seq_len = max_seq_len


class FakeModel:
    """
    Deterministic stand-in for V1LanguageModel.

    Always makes `favorite_token` the argmax, except right after
    `eos_token` has already been generated (irrelevant here since we
    stop the loop) and lets tests control EOS timing via
    `eos_after`.
    """

    def __init__(self, vocab_size, favorite_token, eos_token=None, eos_after=None, max_seq_len=None):
        self.vocab_size = vocab_size
        self.favorite_token = favorite_token
        self.eos_token = eos_token
        self.eos_after = eos_after
        self.config = FakeConfig(max_seq_len=max_seq_len)
        self.calls = 0

    def __call__(self, input_ids):
        self.calls += 1

        B, T = input_ids.shape

        logits = torch.full((B, T, self.vocab_size), -10.0)

        if (
            self.eos_token is not None
            and self.eos_after is not None
            and self.calls > self.eos_after
        ):
            logits[:, -1, self.eos_token] = 10.0
        else:
            logits[:, -1, self.favorite_token] = 10.0

        return logits, None


class FakeTokenizer:
    """Whitespace tokenizer with a fixed vocabulary, plus <eos>."""

    def __init__(self):
        self.vocab = {"hello": 0, "world": 1, "<eos>": 2}
        self.id_to_str = {v: k for k, v in self.vocab.items()}

    def encode(self, text):
        return [self.vocab[word] for word in text.split()]

    def decode(self, token_ids):
        return " ".join(self.id_to_str[i] for i in token_ids)

    def token_to_id(self, token):
        return self.vocab.get(token)


def test_generate_stops_at_max_new_tokens():
    tokenizer = FakeTokenizer()
    model = FakeModel(vocab_size=3, favorite_token=1)
    sampler = Sampler(greedy=True)

    runner = ModelRunner(model, tokenizer, sampler, device="cpu")

    result = runner.generate("hello", max_new_tokens=5)

    assert result["stats"]["generated_tokens"] == 5
    assert result["stats"]["prompt_tokens"] == 1
    assert result["stats"]["total_tokens"] == 6
    assert result["token_ids"] == [0, 1, 1, 1, 1, 1]


def test_generate_stops_at_eos():
    tokenizer = FakeTokenizer()
    model = FakeModel(vocab_size=3, favorite_token=1, eos_token=2, eos_after=2)
    sampler = Sampler(greedy=True)

    runner = ModelRunner(model, tokenizer, sampler, device="cpu")

    result = runner.generate("hello", max_new_tokens=10)

    assert result["token_ids"][-1] == 2
    assert result["stats"]["generated_tokens"] == 3


def test_generate_respects_max_seq_len():
    tokenizer = FakeTokenizer()
    model = FakeModel(vocab_size=3, favorite_token=1, max_seq_len=3)
    sampler = Sampler(greedy=True)

    runner = ModelRunner(model, tokenizer, sampler, device="cpu")

    result = runner.generate("hello", max_new_tokens=10)

    assert result["stats"]["total_tokens"] <= 3


def test_generate_decodes_text():
    tokenizer = FakeTokenizer()
    model = FakeModel(vocab_size=3, favorite_token=1)
    sampler = Sampler(greedy=True)

    runner = ModelRunner(model, tokenizer, sampler, device="cpu")

    result = runner.generate("hello", max_new_tokens=2)

    assert result["text"] == "hello world world"


def test_generate_recomputes_full_sequence_each_step():
    """Naive Phase 3 loop: no KV cache, sequence length grows by 1 each call."""

    tokenizer = FakeTokenizer()
    model = FakeModel(vocab_size=3, favorite_token=1)
    sampler = Sampler(greedy=True)

    runner = ModelRunner(model, tokenizer, sampler, device="cpu")

    runner.generate("hello", max_new_tokens=3)

    assert model.calls == 3


def test_generate_works_with_stochastic_sampling():
    torch.manual_seed(0)

    tokenizer = FakeTokenizer()
    model = FakeModel(vocab_size=3, favorite_token=1)
    sampler = Sampler(temperature=1.0)

    runner = ModelRunner(model, tokenizer, sampler, device="cpu")

    result = runner.generate("hello", max_new_tokens=5)

    assert result["stats"]["generated_tokens"] == 5
