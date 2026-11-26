from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import asdict, dataclass, is_dataclass
from itertools import cycle
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
import yaml
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, random_split

from model.architecture import GirivinityConfig, GirivinityModel
from app.llm.girivinity_tokenizer import GirivinityTokenizer

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PretrainConfig:
    model_config: GirivinityConfig
    batch_size: int
    learning_rate: float
    warmup_steps: int
    max_steps: int
    gradient_accumulation_steps: int
    checkpoint_every: int
    data_path: str
    output_dir: str
    train_seq_len: int = 4096
    use_amp: bool = True
    use_fsdp: bool = False
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be at least 1")
        if self.checkpoint_every < 1:
            raise ValueError("checkpoint_every must be at least 1")
        if not (2 <= self.train_seq_len <= self.model_config.max_seq_len):
            raise ValueError(
                f"train_seq_len ({self.train_seq_len}) must be between 2 and model_config.max_seq_len ({self.model_config.max_seq_len})"
            )

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "PretrainConfig":
        raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        pretrain = (raw.get("training", {}) or {}).get("pretrain", {}) or {}
        if not pretrain:
            raise ValueError("Missing training.pretrain section in config")

        model_config_raw = pretrain.get("model_config") or _model_config_from_root(raw)
        model_cfg = GirivinityConfig(**model_config_raw)
        train_seq_len = int(pretrain.get("train_seq_len", model_cfg.max_seq_len))
        return cls(
            model_config=model_cfg,
            batch_size=int(pretrain["batch_size"]),
            learning_rate=float(pretrain["learning_rate"]),
            warmup_steps=int(pretrain["warmup_steps"]),
            max_steps=int(pretrain["max_steps"]),
            gradient_accumulation_steps=int(pretrain["gradient_accumulation_steps"]),
            checkpoint_every=int(pretrain["checkpoint_every"]),
            data_path=str(pretrain["data_path"]),
            output_dir=str(pretrain["output_dir"]),
            train_seq_len=train_seq_len,
            use_amp=bool(pretrain.get("use_amp", True)),
            use_fsdp=bool(pretrain.get("use_fsdp", False)),
            device=str(pretrain.get("device", raw.get("compute", {}).get("device", "auto"))),
        )


class _PackedTokenDataset(Dataset[dict[str, Tensor]]):
    def __init__(self, token_ids: list[int] | Tensor, seq_len: int) -> None:
        if seq_len < 2:
            raise ValueError("train_seq_len must be at least 2")
        num_tokens = len(token_ids) if isinstance(token_ids, list) else token_ids.numel()
        if num_tokens < seq_len + 1:
            raise ValueError(
                f"Not enough tokens ({num_tokens}) to build one packed sequence of length {seq_len}"
            )

        usable_tokens = ((num_tokens - 1) // seq_len) * seq_len + 1
        if isinstance(token_ids, list):
            self._tokens = torch.tensor(token_ids[:usable_tokens], dtype=torch.long)
        else:
            self._tokens = token_ids[:usable_tokens].to(dtype=torch.long)
        self._seq_len = seq_len
        self._num_sequences = (usable_tokens - 1) // seq_len

    def __len__(self) -> int:
        return self._num_sequences

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        start = idx * self._seq_len
        end = start + self._seq_len
        return {
            "input_ids": self._tokens[start:end],
            "labels": self._tokens[start + 1 : end + 1],
        }


def _validate_manifest_and_schema(record: dict, line_no: int) -> None:
    """Validate JSONL schema preserving document provenance, licence, boundaries."""
    if not isinstance(record, dict):
        raise ValueError(f"Line {line_no}: record must be a JSON object")


def _encode_single_doc(tokenizer: GirivinityTokenizer, text: str, eos_id: int | None) -> list[int]:
    """Encode a single text document ensuring exactly one EOS terminates it and no duplicate BOS/EOS."""
    raw_tok = getattr(tokenizer, "tokenizer", None)
    # Check if underlying tokenizer has a post_processor that automatically appends EOS
    encoded = tokenizer.encode(text)
    if not encoded:
        return []
    # If the encoded output already ends with eos_id, do not append another EOS
    if eos_id is not None:
        if encoded[-1] != eos_id:
            encoded.append(eos_id)
    return encoded


class StreamingPackedDataset(Dataset[dict[str, Tensor]]):
    """Memory-efficient, document-isolated, source-level split packed token dataset."""

    def __init__(self, doc_token_lists: list[list[int]], seq_len: int) -> None:
        self.seq_len = seq_len
        flat_tokens: list[int] = []
        for doc in doc_token_lists:
            flat_tokens.extend(doc)

        if len(flat_tokens) < seq_len + 1:
            self._num_sequences = 0
            self._tokens = torch.empty(0, dtype=torch.long)
        else:
            usable = ((len(flat_tokens) - 1) // seq_len) * seq_len + 1
            self._tokens = torch.tensor(flat_tokens[:usable], dtype=torch.long)
            self._num_sequences = (usable - 1) // seq_len

    def __len__(self) -> int:
        return self._num_sequences

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        start = idx * self.seq_len
        end = start + self.seq_len
        return {
            "input_ids": self._tokens[start:end],
            "labels": self._tokens[start + 1 : end + 1],
        }


class GirivinityPretrainer:
    def __init__(self, config: PretrainConfig) -> None:
        self.config = config
        self.device = _resolve_device(config.device)
        self.on_gpu = self.device.type == "cuda"
        _log_device(self.on_gpu)
        self.model = GirivinityModel(config.model_config)
        self.model.gradient_checkpointing_enable()
        self.model.to(self.device)
        self._wrap_fsdp_if_requested()

        self.train_loader: DataLoader | None = None
        self.val_loader: DataLoader | None = None
        self._base_model = self._unwrap_model(self.model)

    def load_data(self) -> DataLoader:
        tokenizer = self._load_tokenizer()

        # Enforce tokenizer and model vocabulary equality
        tok_vocab_size = len(tokenizer)
        model_vocab_size = self.config.model_config.vocab_size
        if tok_vocab_size != model_vocab_size:
            raise ValueError(
                f"Tokenizer vocabulary size ({tok_vocab_size}) does not match model vocab_size ({model_vocab_size})."
            )

        eos_id = _token_id(tokenizer, "[EOS]")
        train_seq_len = self.config.train_seq_len

        # Stream documents from JSONL and group tokens by document to allow source-level train/val split
        doc_tokens: list[list[int]] = []
        path = Path(self.config.data_path)
        if not path.exists():
            raise FileNotFoundError(f"Training data not found: {path}")

        with path.open(encoding="utf-8") as src:
            for line_no, line in enumerate(src, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
                _validate_manifest_and_schema(record, line_no)
                texts = _extract_texts(record)
                for t in texts:
                    encoded = _encode_single_doc(tokenizer, t, eos_id)
                    if encoded:
                        # Validate all token IDs in range
                        if any(tid < 0 or tid >= model_vocab_size for tid in encoded):
                            raise ValueError(f"Line {line_no}: Token ID out of range [0, {model_vocab_size})")
                        doc_tokens.append(encoded)

        if not doc_tokens:
            raise ValueError(f"No valid training text found in {self.config.data_path}")

        # Split train and validation by document/source BEFORE packing
        num_docs = len(doc_tokens)
        val_doc_count = max(1, int(num_docs * 0.1)) if num_docs >= 5 else 0

        if val_doc_count > 0:
            # Deterministic split based on document indices
            val_docs = doc_tokens[:val_doc_count]
            train_docs = doc_tokens[val_doc_count:]
        else:
            train_docs = doc_tokens
            val_docs = []

        train_dataset = StreamingPackedDataset(train_docs, seq_len=train_seq_len)
        if len(train_dataset) == 0:
            raise ValueError(
                f"Corpus too small: cannot build a single training sequence of length {train_seq_len} tokens."
            )

        val_dataset = StreamingPackedDataset(val_docs, seq_len=train_seq_len) if val_docs else None

        num_workers = 4 if self.on_gpu else 0
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=self.on_gpu,
        )
        if val_dataset and len(val_dataset) > 0:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=num_workers,
                pin_memory=self.on_gpu,
            )
        else:
            self.val_loader = None

        return self.train_loader

    def train(self) -> None:
        train_loader = self.train_loader or self.load_data()
        optimizer = AdamW(self.model.parameters(), lr=self.config.learning_rate)
        scheduler = LambdaLR(optimizer, lr_lambda=self._lr_multiplier)
        device = self.device
        on_gpu = device.type == "cuda"
        amp_enabled = self.config.use_amp and on_gpu
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

        self.model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        running_tokens = 0
        window_started = time.perf_counter()
        accumulated_microsteps = 0
        optimizer_step = 0
        checkpoint_saved = False

        # Explicit epoch loop rather than itertools.cycle(train_loader)
        # This refreshes loader shuffle and sampler order deterministically each epoch
        epoch = 1
        while optimizer_step < self.config.max_steps:
            for batch in train_loader:
                loss, token_count = self._training_step(batch, amp_enabled)
                scaled_loss = loss / self.config.gradient_accumulation_steps
                scaler.scale(scaled_loss).backward()
                running_loss += loss.detach().item()
                running_tokens += token_count
                accumulated_microsteps += 1

                if accumulated_microsteps % self.config.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    optimizer_step += 1
                    elapsed = max(time.perf_counter() - window_started, 1e-9)
                    current_loss = running_loss / self.config.gradient_accumulation_steps
                    tps = running_tokens / elapsed
                    lr = scheduler.get_last_lr()[0]
                    print(f"step={optimizer_step} loss={current_loss:.4f} lr={lr:.2e} tokens/sec={tps:.0f}")

                    checkpoint_saved = False
                    if optimizer_step % self.config.checkpoint_every == 0:
                        self._save_checkpoint(optimizer_step)
                        checkpoint_saved = True

                    running_loss = 0.0
                    running_tokens = 0
                    window_started = time.perf_counter()

                    if optimizer_step >= self.config.max_steps:
                        break

            # Handle partial accumulation remainder at epoch boundary cleanly
            remainder = accumulated_microsteps % self.config.gradient_accumulation_steps
            if remainder != 0 and optimizer_step < self.config.max_steps:
                # Rescale gradients that were scaled by 1/gradient_accumulation_steps
                scaler.unscale_(optimizer)
                for p in self.model.parameters():
                    if p.grad is not None:
                        p.grad.data.mul_(self.config.gradient_accumulation_steps / remainder)
                clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                running_loss = 0.0
                running_tokens = 0
                accumulated_microsteps = 0
                window_started = time.perf_counter()

            epoch += 1

        if not checkpoint_saved:
            self._save_checkpoint(optimizer_step)
        if on_gpu:
            torch.cuda.empty_cache()

    @torch.no_grad()
    def estimate_loss(self) -> dict[str, float]:
        if self.train_loader is None or self.val_loader is None:
            self.load_data()
        assert self.train_loader is not None
        assert self.val_loader is not None

        was_training = self.model.training
        self.model.eval()
        losses = {
            "train": self._loader_loss(self.train_loader),
            "val": self._loader_loss(self.val_loader),
        }
        if was_training:
            self.model.train()
        return losses

    def _training_step(self, batch: dict[str, Tensor], amp_enabled: bool) -> tuple[Tensor, int]:
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        labels = batch["labels"].to(self.device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            output = self.model(input_ids)
            if len(output) == 3:
                # MTP-enabled model: forward() returns
                # (main_logits, mtp_logits_list, kv_caches).
                main_logits, mtp_logits_list, _ = output
                loss = F.cross_entropy(
                    main_logits.reshape(-1, main_logits.size(-1)),
                    labels.reshape(-1),
                )
                mtp_weight = getattr(self.model, "cfg", None)
                weight = mtp_weight.mtp_loss_weight if mtp_weight is not None else 0.3
                for depth, mtp_logits in enumerate(mtp_logits_list, start=1):
                    if mtp_logits.size(1) <= depth:
                        # Sequence too short for this depth's shift to
                        # produce any positions; skip rather than compute
                        # cross-entropy over an empty tensor (which can
                        # silently yield nan instead of raising).
                        continue
                    shifted_logits = mtp_logits[:, :-depth, :]
                    shifted_labels = labels[:, depth:]
                    loss = loss + weight * F.cross_entropy(
                        shifted_logits.reshape(-1, shifted_logits.size(-1)),
                        shifted_labels.reshape(-1),
                    )
            else:
                logits, _ = output
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                )
            # Router z-loss needs NO code here: it is injected inside
            # GirivinityModel.forward (value-neutral gradient patch) so it
            # works with the gradient checkpointing enabled above.
        return loss, labels.numel()

    def _loader_loss(self, loader: DataLoader) -> float:
        total_loss = 0.0
        total_batches = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)
            output = self.model(input_ids)
            logits = output[0]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            total_loss += loss.item()
            total_batches += 1
        return total_loss / max(total_batches, 1)

    def _lr_multiplier(self, step: int) -> float:
        if self.config.warmup_steps > 0 and step < self.config.warmup_steps:
            return max((step + 1) / self.config.warmup_steps, 1e-8)
        decay_steps = max(self.config.max_steps - self.config.warmup_steps, 1)
        progress = min(max((step - self.config.warmup_steps) / decay_steps, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def _save_checkpoint(self, step: int) -> None:
        checkpoint_dir = Path(self.config.output_dir) / f"step-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._save_model(checkpoint_dir)
        (checkpoint_dir / "pretrain_config.json").write_text(
            json.dumps(_dataclass_to_dict(self.config), indent=2),
            encoding="utf-8",
        )

    def _save_final_artifacts(self) -> None:
        """Save canonical final inference and fine-tuning artifact layout.

        Saves to output_dir/final:
        - model.pt
        - config.yaml
        - tokenizer.json (copied from loaded tokenizer if available)
        - manifest_metadata.json
        """
        final_dir = Path(self.config.output_dir) / "final"
        final_dir.mkdir(parents=True, exist_ok=True)

        # Save model weights atomically
        tmp_weights = final_dir / "model.pt.tmp"
        torch.save(self._base_model.state_dict(), tmp_weights)
        if (final_dir / "model.pt").exists():
            (final_dir / "model.pt").unlink()
        tmp_weights.rename(final_dir / "model.pt")

        # Save architecture config.yaml
        cfg_dict = asdict(self.config.model_config)
        (final_dir / "config.yaml").write_text(yaml.dump(cfg_dict), encoding="utf-8")

        # Copy/save tokenizer.json if available
        try:
            tokenizer = self._load_tokenizer()
            if hasattr(tokenizer, "save"):
                tokenizer.save(str(final_dir))
        except Exception as exc:
            logger.warning("Could not package tokenizer into final checkpoint: %s", exc)

        metadata = {
            "version": "1.0",
            "format": "canonical_pt_yaml",
            "vocab_size": self.config.model_config.vocab_size,
            "max_seq_len": self.config.model_config.max_seq_len,
            "train_seq_len": self.config.train_seq_len,
            "saved_at": time.time(),
        }
        (final_dir / "checkpoint_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def _save_model(self, checkpoint_dir: Path) -> None:
        if _is_fsdp_model(self.model):
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
                state_dict = self.model.state_dict()
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                (checkpoint_dir / "config.json").write_text(
                    json.dumps(asdict(self.config.model_config), indent=2),
                    encoding="utf-8",
                )
                torch.save(state_dict, checkpoint_dir / "pytorch_model.bin")
            return

        self._base_model.save_pretrained(str(checkpoint_dir))

    def _load_tokenizer(self) -> GirivinityTokenizer:
        for path in _candidate_tokenizer_paths(self.config.data_path, self.config.output_dir):
            try:
                return GirivinityTokenizer.from_file(str(path))
            except FileNotFoundError:
                continue
        candidates = ", ".join(
            str(path)
            for path in _candidate_tokenizer_paths(self.config.data_path, self.config.output_dir)
        )
        raise FileNotFoundError(f"Girivinity tokenizer not found. Tried: {candidates}")

    def _wrap_fsdp_if_requested(self) -> None:
        if not self.config.use_fsdp:
            return
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            raise RuntimeError(
                "use_fsdp=True requested but torch.distributed is not initialized! "
                "Refusing to silently fall back to single-node training."
            )
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        self.model = FSDP(self.model)

    @staticmethod
    def _unwrap_model(model: torch.nn.Module) -> GirivinityModel:
        return model.module if hasattr(model, "module") else model  # type: ignore[return-value]


def _resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("compute.device is 'cuda' but CUDA is not available")
    if requested not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {requested}")
    return torch.device(requested)


def compute_model_parameter_profile(cfg: GirivinityConfig) -> dict[str, Any]:
    """Allocation-free parameter-count and memory profile derivation from config formulas."""
    d = cfg.dim
    V = cfg.vocab_size
    L = cfg.n_layers
    h_q = cfg.n_heads
    h_kv = cfg.n_kv_heads
    head_dim = d // h_q
    mlp_mult = cfg.ffn_multiplier or 2.667
    ffn_dim = int(d * mlp_mult)

    # 1. Embeddings (tied or untied)
    embed_params = V * d
    # 2. Per layer dense attention: Q, K, V, O projections
    q_proj = d * (h_q * head_dim)
    k_proj = d * (h_kv * head_dim)
    v_proj = d * (h_kv * head_dim)
    o_proj = (h_q * head_dim) * d
    attn_layer_params = q_proj + k_proj + v_proj + o_proj
    # 3. Norms per layer: input norm + post-attention norm
    norm_params = 2 * d

    # 4. Feedforward / MoE
    if cfg.moe_enabled:
        routed_exp = cfg.n_routed_experts
        top_k = cfg.n_activated_experts
        shared_exp = cfg.n_shared_experts
        # SwiGLU expert: gate, up, down projections
        single_expert = 3 * d * (ffn_dim // 4 if hasattr(cfg, "moe_intermediate_size") and cfg.moe_intermediate_size else ffn_dim)
        moe_gate = d * routed_exp
        total_moe_layer = moe_gate + (routed_exp + shared_exp) * single_expert
        active_moe_layer = moe_gate + (top_k + shared_exp) * single_expert
    else:
        dense_ffn = 3 * d * ffn_dim
        total_moe_layer = dense_ffn
        active_moe_layer = dense_ffn

    total_params = embed_params + L * (attn_layer_params + norm_params + total_moe_layer)
    active_params_per_token = embed_params + L * (attn_layer_params + norm_params + active_moe_layer)

    # Memory lower bounds (FP32 weights = 4 bytes, AdamW optimizer = 8 bytes, grads = 4 bytes)
    weight_bytes_fp32 = total_params * 4
    grad_bytes_fp32 = total_params * 4
    adamw_bytes = total_params * 8
    total_static_memory_gb = (weight_bytes_fp32 + grad_bytes_fp32 + adamw_bytes) / (1024 ** 3)

    return {
        "total_parameters": total_params,
        "active_parameters_per_token": active_params_per_token,
        "weight_memory_fp32_gb": round(weight_bytes_fp32 / (1024 ** 3), 2),
        "optimizer_memory_adamw_gb": round(adamw_bytes / (1024 ** 3), 2),
        "gradient_memory_fp32_gb": round(grad_bytes_fp32 / (1024 ** 3), 2),
        "total_static_training_gb": round(total_static_memory_gb, 2),
        "note": "Excludes dynamic activations, KV caches, runtime buffers, communication buffers.",
    }


def preflight_feasibility_guard(cfg: GirivinityConfig, available_vram_gb: float | None = None) -> None:
    """Preflight check: refuses training if memory/sharding plan is insufficient."""
    profile = compute_model_parameter_profile(cfg)
    static_gb = profile["total_static_training_gb"]

    if available_vram_gb is not None:
        if available_vram_gb < static_gb:
            raise RuntimeError(
                f"Preflight Feasibility Rejection: Model requires at least {static_gb:.1f} GB VRAM for static states "
                f"(weights + grads + optimizer states), but only {available_vram_gb:.1f} GB is available. "
                f"Sharding / FSDP or multi-node setup is required."
            )


def _log_device(on_gpu: bool) -> None:
    if on_gpu:
        logger.info("Training on GPU: %s", torch.cuda.get_device_name(0))
    else:
        logger.info("Training on CPU — this will be slow, consider a GPU instance")


def _iter_jsonl_texts(data_path: str | Path) -> Iterable[str]:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Training data not found: {path}")

    with path.open(encoding="utf-8") as src:
        for line_no, line in enumerate(src, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
            yield from _extract_texts(record)


def _extract_texts(record: Any) -> list[str]:
    if isinstance(record, str):
        return [record.strip()] if record.strip() else []
    if isinstance(record, list):
        texts: list[str] = []
        for item in record:
            texts.extend(_extract_texts(item))
        return texts
    if not isinstance(record, dict):
        return []

    preferred_keys = (
        "instruction",
        "question",
        "prompt",
        "input",
        "context",
        "response",
        "answer",
        "completion",
        "output",
        "text",
        "code",
    )
    texts = [str(record[key]).strip() for key in preferred_keys if record.get(key)]

    messages = record.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and message.get("content"):
                texts.append(str(message["content"]).strip())

    sources = record.get("sources")
    if isinstance(sources, list):
        for source in sources:
            if isinstance(source, dict):
                for key in ("title", "text", "snippet", "content"):
                    if source.get(key):
                        texts.append(str(source[key]).strip())

    return [text for text in texts if text]


def _candidate_tokenizer_paths(data_path: str, output_dir: str) -> list[Path]:
    data_parent = Path(data_path).parent
    return [
        data_parent,
        data_parent / "tokenizer",
        Path(output_dir),
        Path(output_dir) / "tokenizer",
        Path("girivinity_tokenizer"),
        Path("models/tokenizer"),
        Path("models/base"),
    ]


def _token_id(tokenizer: GirivinityTokenizer, token: str) -> int | None:
    raw_tokenizer = getattr(tokenizer, "tokenizer", None)
    if raw_tokenizer is not None and hasattr(raw_tokenizer, "token_to_id"):
        return raw_tokenizer.token_to_id(token)
    return None


def _model_config_from_root(raw: dict[str, Any]) -> dict[str, Any]:
    model = raw.get("model", {}) or {}
    architecture = model.get("architecture", {}) or {}
    if architecture:
        return architecture
    key_map = {
        "vocab_size": "vocab_size",
        "hidden_dim": "dim",
        "dim": "dim",
        "num_heads": "n_heads",
        "n_heads": "n_heads",
        "num_layers": "n_layers",
        "n_layers": "n_layers",
        "max_seq_len": "max_seq_len",
        "num_kv_heads": "n_kv_heads",
        "n_kv_heads": "n_kv_heads",
        "norm_eps": "norm_eps",
        "rope_theta": "rope_theta",
    }
    result = {target: model[source] for source, target in key_map.items() if source in model}
    # model/architecture.py's GirivinityConfig has no direct ffn_dim field —
    # it derives ffn_dim from ffn_multiplier. Convert if a raw ffn_dim was given.
    if "ffn_dim" in model and "dim" in result:
        result["ffn_multiplier"] = model["ffn_dim"] / result["dim"]
    elif "ffn_dim" in model:
        result["ffn_multiplier"] = model["ffn_dim"] / model.get("hidden_dim", model.get("dim", 3072))
    return result


def _dataclass_to_dict(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _dataclass_to_dict(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: _dataclass_to_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_dataclass_to_dict(item) for item in value]
    return value


def _is_fsdp_model(model: torch.nn.Module) -> bool:
    return model.__class__.__name__ == "FullyShardedDataParallel"


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain GirivinityModel with pure PyTorch")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    config = PretrainConfig.from_yaml(args.config)
    GirivinityPretrainer(config).train()


if __name__ == "__main__":
    main()
