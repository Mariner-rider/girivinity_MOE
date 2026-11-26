"""Deterministic parity test suite for Attention KV-cache and causal masking.

Validates that full-sequence forward passes produce numerically identical logits
to prompt prefill followed by token-by-token cached decoding across varied batch
sizes, sequence lengths, cache lengths, architecture configurations, and padding masks.

Documented tolerance:
- FP32: atol = 1e-5, rtol = 1e-5 (actual observed delta is typically machine epsilon ~1e-7).
"""

import pytest
import torch
from model.architecture import GirivinityConfig, GirivinityModel

DOCUMENTED_FP32_TOLERANCE = 1e-5


def _create_test_model(
    mla_enabled: bool = False,
    kv_sharing_start_layer: int | None = None,
    seed: int = 42,
) -> tuple[GirivinityModel, GirivinityConfig]:
    torch.manual_seed(seed)
    cfg = GirivinityConfig(
        dim=64,
        n_layers=3,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=500,
        max_seq_len=256,
        ffn_multiplier=2.0,
        mla_enabled=mla_enabled,
        kv_sharing_start_layer=kv_sharing_start_layer,
    )
    model = GirivinityModel(cfg)
    model.eval()
    return model, cfg


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize(
    "seq_len,prefill_len",
    [
        (8, 1),   # Minimal prefill (1 token), 7 incremental decode steps
        (16, 4),  # Standard prompt prefill (4 tokens), 12 decode steps
        (24, 12), # Long prefill (12 tokens), 12 decode steps
    ],
)
def test_gqa_full_vs_cached_parity(batch_size: int, seq_len: int, prefill_len: int):
    """Verifies that GQA incremental cached decoding exactly matches full forward logits at every position."""
    model, cfg = _create_test_model(mla_enabled=False, seed=batch_size * 100 + seq_len)
    torch.manual_seed(1234 + prefill_len)
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        full_logits, _ = model(input_ids)

        # Prefill prompt
        prefill_ids = input_ids[:, :prefill_len]
        prefill_logits, kv_caches = model(prefill_ids)

        chunks = [prefill_logits]
        # Token-by-token cached decoding
        for t in range(prefill_len, seq_len):
            step_token = input_ids[:, t : t + 1]
            step_logits, kv_caches = model(step_token, kv_caches=kv_caches)
            chunks.append(step_logits)

        cached_logits = torch.cat(chunks, dim=1)

    assert full_logits.shape == cached_logits.shape
    max_diff = (full_logits - cached_logits).abs().max().item()
    assert (
        max_diff < DOCUMENTED_FP32_TOLERANCE
    ), f"Parity failed: max abs difference {max_diff} exceeds {DOCUMENTED_FP32_TOLERANCE}"
    assert torch.allclose(full_logits, cached_logits, atol=DOCUMENTED_FP32_TOLERANCE, rtol=1e-5)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seq_len,prefill_len", [(12, 3), (16, 8)])
def test_mla_full_vs_cached_parity(batch_size: int, seq_len: int, prefill_len: int):
    """Verifies that MultiHeadLatentAttention (MLA) cached decoding matches full forward."""
    model, cfg = _create_test_model(mla_enabled=True, seed=2000 + batch_size)
    torch.manual_seed(5678 + prefill_len)
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        full_logits, _ = model(input_ids)

        prefill_ids = input_ids[:, :prefill_len]
        prefill_logits, kv_caches = model(prefill_ids)

        chunks = [prefill_logits]
        for t in range(prefill_len, seq_len):
            step_token = input_ids[:, t : t + 1]
            step_logits, kv_caches = model(step_token, kv_caches=kv_caches)
            chunks.append(step_logits)

        cached_logits = torch.cat(chunks, dim=1)

    max_diff = (full_logits - cached_logits).abs().max().item()
    assert (
        max_diff < DOCUMENTED_FP32_TOLERANCE
    ), f"MLA parity failed: max abs diff {max_diff} exceeds {DOCUMENTED_FP32_TOLERANCE}"
    assert torch.allclose(full_logits, cached_logits, atol=DOCUMENTED_FP32_TOLERANCE, rtol=1e-5)


def test_kv_sharing_full_vs_cached_parity():
    """Verifies that cross-layer shared KV cache maintains exact causal alignment."""
    model, cfg = _create_test_model(kv_sharing_start_layer=1, seed=3000)
    torch.manual_seed(3001)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 16))

    with torch.no_grad():
        full_logits, _ = model(input_ids)

        prefill_logits, kv_caches = model(input_ids[:, :6])
        chunks = [prefill_logits]
        for t in range(6, 16):
            step_token = input_ids[:, t : t + 1]
            step_logits, kv_caches = model(step_token, kv_caches=kv_caches)
            chunks.append(step_logits)

        cached_logits = torch.cat(chunks, dim=1)

    max_diff = (full_logits - cached_logits).abs().max().item()
    assert max_diff < DOCUMENTED_FP32_TOLERANCE
    assert torch.allclose(full_logits, cached_logits, atol=DOCUMENTED_FP32_TOLERANCE, rtol=1e-5)


def test_multi_token_chunk_decoding_parity():
    """Verifies chunk verification passes where Q_len > 1 and K_len > Q_len (e.g. speculative decoding)."""
    model, cfg = _create_test_model(seed=4000)
    torch.manual_seed(4001)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 16))

    with torch.no_grad():
        full_logits, _ = model(input_ids)

        # Prefill 6 tokens
        prefill_logits, kv_caches = model(input_ids[:, :6])
        # Chunk 1: tokens 6..9 (length 3)
        chunk1_logits, kv_caches = model(input_ids[:, 6:9], kv_caches=kv_caches)
        # Chunk 2: tokens 9..14 (length 5)
        chunk2_logits, kv_caches = model(input_ids[:, 9:14], kv_caches=kv_caches)
        # Token-by-token: tokens 14..16 (length 1 each)
        tok14_logits, kv_caches = model(input_ids[:, 14:15], kv_caches=kv_caches)
        tok15_logits, kv_caches = model(input_ids[:, 15:16], kv_caches=kv_caches)

        stitched_logits = torch.cat(
            [prefill_logits, chunk1_logits, chunk2_logits, tok14_logits, tok15_logits],
            dim=1,
        )

    max_diff = (full_logits - stitched_logits).abs().max().item()
    assert max_diff < DOCUMENTED_FP32_TOLERANCE
    assert torch.allclose(full_logits, stitched_logits, atol=DOCUMENTED_FP32_TOLERANCE, rtol=1e-5)


def test_batched_padding_mask_parity():
    """Verifies causal alignment under batched padding with attention_mask."""
    model, cfg = _create_test_model(seed=5000)
    torch.manual_seed(5001)
    # Batch size 2: sequence 0 has length 12 (no padding), sequence 1 has 3 pad tokens at left
    # Valid mask: 1 for active tokens, 0 for pad
    seq_len = 12
    input_ids = torch.randint(1, cfg.vocab_size, (2, seq_len))
    # Pad first 3 positions of sequence 1
    input_ids[1, :3] = 0
    attention_mask = torch.ones((2, seq_len), dtype=torch.bool)
    attention_mask[1, :3] = False

    with torch.no_grad():
        full_logits, _ = model(input_ids, attention_mask=attention_mask)

        # Prefill 6 tokens
        prefill_mask = attention_mask[:, :6]
        prefill_logits, kv_caches = model(input_ids[:, :6], attention_mask=prefill_mask)

        chunks = [prefill_logits]
        for t in range(6, seq_len):
            step_mask = attention_mask[:, : t + 1]
            step_logits, kv_caches = model(
                input_ids[:, t : t + 1],
                kv_caches=kv_caches,
                attention_mask=step_mask,
            )
            chunks.append(step_logits)

        cached_logits = torch.cat(chunks, dim=1)

    # For sequence 0 (all valid): all positions must match
    diff_seq0 = (full_logits[0] - cached_logits[0]).abs().max().item()
    assert diff_seq0 < DOCUMENTED_FP32_TOLERANCE

    # For sequence 1: unpadded positions (index 3 onwards) must match
    diff_seq1 = (full_logits[1, 3:] - cached_logits[1, 3:]).abs().max().item()
    assert diff_seq1 < DOCUMENTED_FP32_TOLERANCE


def test_generate_parity_cached_vs_uncached():
    """Verifies that generate() with use_cache=True produces identical output to use_cache=False."""
    model, cfg = _create_test_model(seed=6000)
    torch.manual_seed(6001)
    prompt = torch.randint(0, cfg.vocab_size, (1, 8))

    gen_uncached = model.generate(prompt, max_new_tokens=10, temperature=0.0, use_cache=False)
    gen_cached = model.generate(prompt, max_new_tokens=10, temperature=0.0, use_cache=True)

    assert torch.equal(
        gen_uncached, gen_cached
    ), "Greedy generation with use_cache=True differed from use_cache=False!"
