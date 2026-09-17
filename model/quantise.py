from __future__ import annotations

import argparse
import json
import logging
import subprocess
from pathlib import Path

import torch

from model.architecture import GirivinityConfig, GirivinityModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

TOKENISER_PATH = "models/tokeniser/tokeniser.json"


def _load_model_and_config(weights_path: str):
    weights = Path(weights_path)
    if not (weights / "model.pt").exists():
        raise FileNotFoundError(f"no checkpoint at {weights}/model.pt")

    # Prefer the architecture config saved alongside the checkpoint by
    # model/train.py; fall back to the root config.yaml.
    cfg = None
    cfg_yaml = weights / "config.yaml"
    if cfg_yaml.exists():
        import yaml

        data = yaml.safe_load(cfg_yaml.read_text()) or {}
        fields = {k: v for k, v in data.items() if k in GirivinityConfig.__dataclass_fields__}
        cfg = GirivinityConfig(**fields)
    if cfg is None:
        cfg = GirivinityConfig.from_yaml()

    unsupported = [
        name
        for name, enabled in (
            ("moe_enabled", cfg.moe_enabled),
            ("mla_enabled", cfg.mla_enabled),
            ("mtp_enabled", cfg.mtp_enabled),
            ("ple_enabled", cfg.ple_enabled),
            ("mhc_enabled", cfg.mhc_enabled),
        )
        if enabled
    ]
    if unsupported:
        raise SystemExit(
            "GGUF export currently supports the dense GQA+SwiGLU configuration only. "
            f"Active unsupported feature(s): {', '.join(unsupported)}. Run inference via "
            "the PyTorch path (model/inference.py) for this checkpoint."
        )

    model = GirivinityModel(cfg)
    state = torch.load(weights / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, cfg


def _llama_tensor_name(name: str):
    """Map Girivinity state_dict names to llama.cpp 'llama' architecture names."""
    if name == "embed.weight":
        return "token_embd.weight"
    if name == "norm.weight":
        return "output_norm.weight"
    if name == "lm_head.weight":
        return None  # tied to token_embd; llama.cpp ties automatically
    parts = name.split(".")
    if parts[0] == "layers" and len(parts) >= 3:
        n = parts[1]
        leaf = ".".join(parts[2:])
        table = {
            "attn_norm.weight": "blk.{n}.attn_norm.weight",
            "ffn_norm.weight": "blk.{n}.ffn_norm.weight",
            "attn.q_proj.weight": "blk.{n}.attn_q.weight",
            "attn.k_proj.weight": "blk.{n}.attn_k.weight",
            "attn.v_proj.weight": "blk.{n}.attn_v.weight",
            "attn.o_proj.weight": "blk.{n}.attn_output.weight",
            "ffn.gate.weight": "blk.{n}.ffn_gate.weight",
            "ffn.up.weight": "blk.{n}.ffn_up.weight",
            "ffn.down.weight": "blk.{n}.ffn_down.weight",
        }
        template = table.get(leaf)
        return template.format(n=n) if template else None
    return None


def _byte_decoder_for(data: dict):
    """GPT-2 byte-level BPE char->byte map (identity for the 188 visible bytes)."""
    visible = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    byte_list = visible[:]
    char_list = [chr(b) for b in visible]
    extra = 0
    for b in range(256):
        if b not in byte_list:
            byte_list.append(b)
            char_list.append(chr(256 + extra))
            extra += 1
    return dict(zip(char_list, byte_list))


def _is_byte_level(data: dict) -> bool:
    pre = data.get("pre_tokenizer") or {}
    if pre.get("type") == "ByteLevel":
        return True
    if pre.get("type") == "Sequence":
        return any(p.get("type") == "ByteLevel" for p in pre.get("pretokenizers", []))
    return False


def _write_tokenizer_metadata(writer, tokeniser_path: str) -> None:
    data = json.loads(Path(tokeniser_path).read_text())
    model = data.get("model", {})
    vocab = model.get("vocab", {})
    n = len(vocab)
    if n == 0:
        raise ValueError(f"{tokeniser_path} has an empty vocab")

    ordered = [""] * n
    for tok, idx in vocab.items():
        if isinstance(idx, int) and 0 <= idx < n:
            ordered[idx] = tok

    byte_decoder = _byte_decoder_for(data) if _is_byte_level(data) else {}

    def to_bytes(tok: str) -> bytes:
        if byte_decoder:
            try:
                return bytes(byte_decoder[c] for c in tok)
            except KeyError:
                pass
        return tok.encode("utf-8", errors="replace")

    token_bytes = [to_bytes(t) for t in ordered]
    token_types = [1] * n  # 1 = NORMAL in the ggml token-type enum

    unk_id, bos_id, eos_id = 0, 1, 2
    for added in data.get("added_tokens", []) or []:
        idx = added.get("id")
        content = (added.get("content") or "").lower()
        special = bool(added.get("special"))
        if isinstance(idx, int) and 0 <= idx < n:
            token_types[idx] = 3 if special else 4  # CONTROL / USER_DEFINED
            if "unk" in content:
                unk_id = idx
            elif content in ("<s>", "<bos>", "[bos]"):
                bos_id = idx
            elif content in ("</s>", "<eos>", "[eos]"):
                eos_id = idx

    writer.add_tokenizer_model("gpt2")  # byte-level BPE path in llama.cpp
    writer.add_token_list(token_bytes)
    writer.add_token_scores([0.0] * n)
    writer.add_token_types(token_types)
    writer.add_unk_token_id(unk_id)
    writer.add_bos_token_id(bos_id)
    writer.add_eos_token_id(eos_id)
    merges = model.get("merges", []) or []
    if merges:
        writer.add_array("tokenizer.ggml.merges", [m.encode("utf-8") for m in merges])


def export_to_gguf(
    weights_path: str,
    output_dir: str,
    quant_type: str = "Q4_K_M",
    tokeniser_path: str = TOKENISER_PATH,
) -> Path:
    import numpy as np
    from gguf import GGUFWriter

    model, cfg = _load_model_and_config(weights_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    effective_seq_len = int(cfg.max_seq_len * cfg.rope_scaling_factor)
    f16_path = out_dir / "model-f16.gguf"
    writer = GGUFWriter(str(f16_path), arch="llama")
    writer.add_architecture()
    writer.add_context_length(effective_seq_len)
    writer.add_embedding_length(cfg.dim)
    writer.add_block_count(cfg.n_layers)
    writer.add_feed_forward_length(cfg.ffn_dim)
    writer.add_head_count(cfg.n_heads)
    writer.add_head_count_kv(cfg.n_kv_heads)
    writer.add_rope_dimension_count(cfg.head_dim)
    writer.add_rope_freq_base(float(cfg.rope_theta))
    writer.add_vocab_size(cfg.vocab_size)
    if hasattr(writer, "add_attention_layer_norm_rms_eps"):
        writer.add_attention_layer_norm_rms_eps(cfg.norm_eps)
    else:
        writer.add_float32("llama.attention.layer_norm_rms_epsilon", cfg.norm_eps)

    if Path(tokeniser_path).exists():
        _write_tokenizer_metadata(writer, tokeniser_path)
    else:
        logger.warning("tokeniser not found at %s — writing weights only", tokeniser_path)

    state = model.state_dict()
    for name, tensor in state.items():
        mapped = _llama_tensor_name(name)
        if mapped is None:
            logger.info("skipping %s (tied or unmapped)", name)
            continue
        arr = tensor.detach().to(torch.float16).numpy().astype(np.float16)
        writer.add_tensor(mapped, arr)
    writer.write()
    logger.info("f16 GGUF written to %s", f16_path)

    quant = quant_type.upper()
    if quant in ("F16", "F32", "BF16"):
        final = f16_path
    else:
        final = out_dir / "model.gguf"
        quantize_bin = Path("llama.cpp/llama-quantize")
        if not quantize_bin.exists():
            quantize_bin = Path("llama.cpp/build/bin/llama-quantize")
        if not quantize_bin.exists():
            raise FileNotFoundError(
                "llama-quantize not found. Build llama.cpp "
                "(cmake -B build && cmake --build build --config Release) or pass --quant F16."
            )
        result = subprocess.run(
            [str(quantize_bin), str(f16_path), str(final), quant],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"llama-quantize failed:\n{result.stderr}")
        logger.info("Quantized GGUF saved to %s", final)

    return final


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="models/base/final")
    parser.add_argument("--output", default="models/girivinity_quantised")
    parser.add_argument("--quant", default="Q4_K_M")
    parser.add_argument("--tokeniser", default=TOKENISER_PATH)
    args = parser.parse_args()
    export_to_gguf(args.weights, args.output, args.quant, args.tokeniser)