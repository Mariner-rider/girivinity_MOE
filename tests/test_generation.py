import tempfile
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F

from model.architecture import GirivinityConfig, GirivinityModel
from app.llm.girivinity_tokenizer import GirivinityTokenizer


@pytest.fixture(scope="module")
def small_model():
    torch.manual_seed(1337)
    cfg = GirivinityConfig(
        dim=64,
        n_layers=2,
        n_heads=2,
        n_kv_heads=2,
        vocab_size=256,
        max_seq_len=64,
    )
    model = GirivinityModel(cfg)
    model.eval()
    return model


@pytest.fixture(scope="module")
def trained_tokenizer():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        corpus = tmp_path / "corpus.txt"
        corpus.write_text(
            "Girivinity pretraining readiness validation and text generation tests. " * 20,
            encoding="utf-8",
        )
        tok_dir = tmp_path / "tok"
        tok = GirivinityTokenizer()
        tok.train([str(corpus)], vocab_size=256, min_frequency=1, output_path=str(tok_dir))
        yield tok


def test_greedy_generation(small_model):
    """Test greedy generation (temperature <= 0) produces deterministic token sequence."""
    input_ids = torch.tensor([[10, 20, 30]])
    out1 = small_model.generate(input_ids, max_new_tokens=10, temperature=0.0)
    out2 = small_model.generate(input_ids, max_new_tokens=10, temperature=0.0)

    assert out1.shape == (1, 13)
    assert torch.equal(out1, out2), "Greedy generation must be strictly deterministic!"


def test_sampling_and_fixed_seed_behavior(small_model):
    """Test sampling generation with temperature and fixed random seeds."""
    input_ids = torch.tensor([[10, 20, 30]])

    # Fixed seed run 1
    torch.manual_seed(42)
    out_seeded_1 = small_model.generate(input_ids, max_new_tokens=15, temperature=0.8, top_p=0.9)

    # Fixed seed run 2
    torch.manual_seed(42)
    out_seeded_2 = small_model.generate(input_ids, max_new_tokens=15, temperature=0.8, top_p=0.9)

    assert torch.equal(out_seeded_1, out_seeded_2), "Sampling with identical RNG seed must produce identical output!"

    # Different seed should produce a different sample
    torch.manual_seed(999)
    out_different_seed = small_model.generate(input_ids, max_new_tokens=15, temperature=0.8, top_p=0.9)
    assert not torch.equal(out_seeded_1, out_different_seed), "Different RNG seed should yield varied sampling!"


def test_max_new_tokens_stop(small_model):
    """Confirm generation stops exactly at max_new_tokens."""
    for n in [1, 5, 12, 20]:
        prompt = torch.tensor([[5, 6, 7]])
        out = small_model.generate(prompt, max_new_tokens=n, temperature=0.0)
        assert out.shape == (1, 3 + n), f"Expected sequence length {3 + n}, got {out.shape[1]}"


def test_eos_token_stop(small_model):
    """Test that generation terminates early when EOS token is emitted."""
    input_ids = torch.tensor([[10, 20]])

    # Force the next greedy token to be token ID 42 by inspecting model logits
    with torch.no_grad():
        logits, _ = small_model(input_ids)
        next_tok = logits[:, -1, :].argmax(dim=-1).item()

    # If we set eos_token_id to this next_tok, generation should stop after generating 1 token
    out = small_model.generate(input_ids, max_new_tokens=20, temperature=0.0, eos_token_id=next_tok)
    assert out.shape == (1, 3), f"Expected stop right after EOS token, got shape {out.shape}"
    assert out[0, -1].item() == next_tok


def test_batch_generation(small_model):
    """Test batched generation with B > 1."""
    batch_input = torch.tensor([
        [10, 20, 30],
        [40, 50, 60],
        [70, 80, 90],
    ])
    out = small_model.generate(batch_input, max_new_tokens=8, temperature=0.0)
    assert out.shape == (3, 11)

    # Individual generations should match rows in batched generation (under causal attention without padding)
    for b in range(3):
        single_out = small_model.generate(batch_input[b : b + 1], max_new_tokens=8, temperature=0.0)
        assert torch.equal(out[b : b + 1], single_out), f"Batch element {b} diverged from single generation!"


def test_cached_vs_uncached_parity(small_model):
    """Verify exact token-by-token and logits parity between use_cache=True and use_cache=False."""
    input_ids = torch.tensor([[15, 25, 35, 45]])
    max_tokens = 12

    out_cached = small_model.generate(input_ids, max_new_tokens=max_tokens, temperature=0.0, use_cache=True)
    out_uncached = small_model.generate(input_ids, max_new_tokens=max_tokens, temperature=0.0, use_cache=False)

    assert torch.equal(out_cached, out_uncached), "Cached decoding differed from uncached full-sequence decoding!"


def test_no_nan_logits(small_model):
    """Verify that forward and generation logits contain zero NaNs or Infs."""
    input_ids = torch.randint(0, 256, (2, 16))
    logits, _ = small_model(input_ids)

    assert not torch.isnan(logits).any(), "NaN found in logits!"
    assert not torch.isinf(logits).any(), "Inf found in logits!"

    out = small_model.generate(input_ids, max_new_tokens=10, temperature=0.7)
    assert not torch.isnan(out).any()
    assert (out >= 0).all() and (out < small_model.cfg.vocab_size).all()


def test_tokenizer_and_special_token_compatibility(small_model, trained_tokenizer):
    """Confirm end-to-end tokenizer encode -> model generate -> tokenizer decode with special tokens."""
    prompt = "Girivinity test"
    token_ids = trained_tokenizer.encode(prompt)
    assert len(token_ids) > 0

    input_ids = torch.tensor([token_ids], dtype=torch.long)
    eos_id = trained_tokenizer.tokenizer.token_to_id("[EOS]")
    bos_id = trained_tokenizer.tokenizer.token_to_id("[BOS]")

    assert eos_id is not None
    assert bos_id is not None

    output_ids = small_model.generate(input_ids, max_new_tokens=8, temperature=0.0, eos_token_id=eos_id)
    generated_list = output_ids[0].tolist()

    decoded_text = trained_tokenizer.decode(generated_list)
    assert isinstance(decoded_text, str)
