from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.architecture import GirivinityConfig, GirivinityModel, MoELayer

logger = logging.getLogger(__name__)

# Every Linear owned by the attention classes in model/architecture.py.
# GroupedQueryAttention:  q_proj / k_proj / v_proj / o_proj
# MultiHeadLatentAttention: w_dq / w_dkv / w_kr / w_uk / w_uv / w_uq / w_qr
# (+ o_proj, shared by both). Names verified against architecture.py.
ATTENTION_TARGETS = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "w_dq", "w_dkv", "w_kr", "w_uk", "w_uv", "w_uq", "w_qr",
)

# Submodules of Expert (used by both routed and shared experts).
EXPERT_SUBMODULES = ("gate", "up", "down")

ADAPTER_FILENAME = "native_lora_adapter.pt"


class LoRALinear(nn.Module):
    """LoRA wrapper around an existing nn.Linear (base weight frozen)."""

    def __init__(self, linear: nn.Linear, r: int = 16, alpha: int = 32,
                 dropout: float = 0.05) -> None:
        super().__init__()
        self.linear = linear          # frozen base weights
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.zeros(r, linear.in_features))
        self.lora_B = nn.Parameter(torch.zeros(linear.out_features, r))
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling


def apply_lora_to_model(model: GirivinityModel, r: int = 16, alpha: int = 32,
                        dropout: float = 0.05, include_experts: bool = True) -> list[str]:
    """Wrap attention projections AND MoE router + routed experts with LoRA.

    Returns the full names of every adapted module.
    """
    adapted: list[str] = []

    def wrap(parent: nn.Module, name: str, full: str) -> None:
        lin = getattr(parent, name)
        if not isinstance(lin, nn.Linear) or isinstance(lin, LoRALinear):
            return  # never double-wrap
        setattr(parent, name, LoRALinear(lin, r, alpha, dropout))
        adapted.append(full)

    for name, module in model.named_modules():
        if isinstance(module, MoELayer):
            if include_experts:
                for i, expert in enumerate(module.routed_experts):
                    for sub in EXPERT_SUBMODULES:
                        wrap(expert, sub, f"{name}.routed_experts.{i}.{sub}")
            wrap(module, "gate", f"{name}.gate")   # router gets LoRA too
        else:
            for t in ATTENTION_TARGETS:
                if hasattr(module, t):
                    wrap(module, t, f"{name}.{t}")
    return adapted


def load_native_model(final_dir: str | Path) -> GirivinityModel:
    """Load a GirivinityModel from the pretrainer's output layout
    (final/model.pt + final/config.yaml) — NOT the config.json +
    pytorch_model.bin layout that GirivinityModel.load_pretrained expects."""
    final_dir = Path(final_dir)
    if not (final_dir / "model.pt").exists():
        raise FileNotFoundError(
            f"Base model not found at {final_dir} — build it first with: "
            f"python -m app.training.pretrain --config config.yaml"
        )

    cfg = GirivinityConfig.from_yaml()
    cfg_path = final_dir / "config.yaml"
    if cfg_path.exists():
        import yaml
        data = yaml.safe_load(cfg_path.read_text()) or {}
        fields = {k: v for k, v in data.items()
                  if k in GirivinityConfig.__dataclass_fields__}
        cfg = GirivinityConfig(**fields)

    model = GirivinityModel(cfg)
    try:
        state = torch.load(final_dir / "model.pt", map_location="cpu",
                           weights_only=True)
    except Exception:
        state = torch.load(final_dir / "model.pt", map_location="cpu",
                           weights_only=False)
    model.load_state_dict(state)
    return model


class NativeSelfTrainer:
    """Native LoRA self-trainer for GirivinityModel — trains attention
    projections AND MoE routed experts + router (PEFT-free, plain PyTorch).

    Drop-in replacement for the PEFT-based loop previously inside
    app/core/self_trainer.py. Adapter is saved as a single
    native_lora_adapter.pt containing only lora_A / lora_B tensors.
    """

    def __init__(
        self,
        base_model_path: str,
        adapter_dir: str,
        lr: float = 2e-4,
        epochs: int = 3,
        max_len: int = 1024,
        batch_size: int = 2,
        r: int = 16,
        alpha: int = 32,
        dropout: float = 0.05,
        include_experts: bool = True,
        tokeniser_path: str = "models/tokeniser/tokeniser.json",
        resume_from: str | None = None,
        gradient_checkpointing: bool = False,
    ) -> None:
        self.model = load_native_model(base_model_path)

        adapted = apply_lora_to_model(
            self.model, r=r, alpha=alpha, dropout=dropout,
            include_experts=include_experts,
        )
        logger.info("LoRA applied to %d modules (include_experts=%s)",
                    len(adapted), include_experts)

        # Freeze base, unfreeze LoRA only.
        for p in self.model.parameters():
            p.requires_grad = False
        for m in self.model.modules():
            if isinstance(m, LoRALinear):
                m.lora_A.requires_grad = True
                m.lora_B.requires_grad = True

        # Resume from a previously saved adapter (same role PEFT's
        # PeftModel.from_pretrained(..., is_trainable=True) played).
        if resume_from:
            ckpt = Path(resume_from) / ADAPTER_FILENAME
            if ckpt.exists():
                prev = torch.load(ckpt, map_location="cpu")
                self.model.load_state_dict(prev, strict=False)
                logger.info("Resumed LoRA adapter from %s", ckpt)

        self.adapter_dir = Path(adapter_dir)
        self.adapter_dir.mkdir(parents=True, exist_ok=True)
        self.lr, self.epochs = lr, epochs
        self.max_len, self.batch_size = max_len, batch_size
        self.tokeniser_path = tokeniser_path
        self.gradient_checkpointing = gradient_checkpointing

        # Device selection, with the same <4GB-free-VRAM CPU fallback the
        # old PEFT loop had.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.on_gpu = device.type == "cuda"
        if self.on_gpu:
            free_vram, _ = torch.cuda.mem_get_info()
            if free_vram < 4 * 1024 ** 3:
                logger.warning("GPU has %.1fGB free (<4GB); falling back to CPU",
                               free_vram / 1e9)
                device = torch.device("cpu")
                self.on_gpu = False
        logger.info("Training on %s",
                    torch.cuda.get_device_name(0) if self.on_gpu else "CPU")
        self.device = device
        self.model.to(self.device)

        if self.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    def train_on_jsonl(self, dataset_path: str) -> float:
        """Train on a JSONL of {"instruction": ..., "response": ...}.
        Returns the average loss of the final epoch (used by the
        loss_abort check in SelfTrainer)."""
        from torch.utils.data import DataLoader
        from model.train import InstructionDataset, collate

        ds = InstructionDataset(dataset_path, self.tokeniser_path, self.max_len)
        if len(ds) == 0:
            raise ValueError(f"Dataset at {dataset_path} is empty")
        loader = DataLoader(
            ds, batch_size=self.batch_size, shuffle=True,
            collate_fn=collate, num_workers=0,
        )
        logger.info("Dataset: %d samples, %d batches/epoch",
                    len(ds), len(loader))

        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=self.lr,
                                weight_decay=0.0, betas=(0.9, 0.95))
        scaler = torch.amp.GradScaler("cuda", enabled=self.on_gpu)

        self.model.train()
        last_epoch_loss = 0.0
        for epoch in range(1, self.epochs + 1):
            total, steps = 0.0, 0
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                with torch.autocast(device_type=self.device.type,
                                    dtype=torch.float16,
                                    enabled=self.on_gpu):
                    out = self.model(x)
                    logits = out[0]   # works for both 2-tuple and MTP 3-tuple
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        y.reshape(-1),
                        ignore_index=-100,
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                total += loss.item()
                steps += 1
            last_epoch_loss = total / max(1, steps)
            logger.info("epoch %d/%d — avg loss: %.4f",
                        epoch, self.epochs, last_epoch_loss)

        adapter_path = self.adapter_dir / ADAPTER_FILENAME
        torch.save(
            {k: v.detach().cpu() for k, v in self.model.state_dict().items()
             if "lora_A" in k or "lora_B" in k},
            adapter_path,
        )
        logger.info("Adapter saved to %s (%d tensors)",
                    adapter_path,
                    sum(1 for k in self.model.state_dict()
                        if "lora_A" in k or "lora_B" in k))
        return last_epoch_loss