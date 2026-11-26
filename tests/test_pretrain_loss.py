#successful with the command : .\.venv\Scripts\python.exe -m pytest tests/test_pretrain_loss.py -v
import os
import json
import shutil
import tempfile
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from model.architecture import GirivinityConfig, GirivinityModel
from app.training.pretrain import PretrainConfig, GirivinityPretrainer
from app.llm.girivinity_tokenizer import GirivinityTokenizer


def _setup_tiny_env(tmp_dir):
    tmp_path = Path(tmp_dir)
    # Train a tiny tokenizer into tmp_dir / "tokenizer"
    corpus_file = tmp_path / "corpus.txt"
    corpus_file.write_text(
        "hello world! this is a synthetic pretraining corpus with multiple tokens for testing. " * 20,
        encoding="utf-8",
    )
    tok_dir = tmp_path / "tokenizer"
    tok = GirivinityTokenizer()
    tok.train([str(corpus_file)], vocab_size=512, min_frequency=1, output_path=str(tok_dir))
    actual_vocab_size = len(tok)

    # Prepare dummy jsonl data
    data_path = tmp_path / "data.jsonl"
    with open(data_path, "w", encoding="utf-8") as f:
        for _ in range(10):
            json.dump({"text": "hello world! this is a synthetic pretraining corpus with multiple tokens for testing."}, f)
            f.write("\n")

    model_cfg = GirivinityConfig(
        dim=64,
        n_layers=2,
        n_heads=2,
        n_kv_heads=2,
        vocab_size=max(512, actual_vocab_size + 64),
        max_seq_len=16,
    )
    pretrain_cfg = PretrainConfig(
        model_config=model_cfg,
        batch_size=2,
        learning_rate=1e-3,
        warmup_steps=0,
        max_steps=10,
        gradient_accumulation_steps=2,
        checkpoint_every=5,
        data_path=str(data_path),
        output_dir=str(tmp_path / "out"),
        use_amp=False,
        use_fsdp=False,
        device="cpu",
    )
    return pretrain_cfg


def test_next_token_shift_and_finite_loss():
    """Verify next-token shift: logits at position t predict token t+1; check finite loss."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg = _setup_tiny_env(tmp_dir)
        pretrainer = GirivinityPretrainer(cfg)
        pretrainer.load_data()
        batch = next(iter(pretrainer.train_loader))

        input_ids = batch["input_ids"]
        labels = batch["labels"]

        # In _PackedTokenDataset: input_ids is tokens[start:end], labels is tokens[start+1:end+1]
        # Verify next-token shift invariant in dataset packing:
        assert input_ids.shape == labels.shape
        # Specifically, labels[b, t] is the token following input_ids[b, t]
        assert torch.equal(input_ids[:, 1:], labels[:, :-1])

        # Execute training step and assert finite non-zero loss
        loss, token_count = pretrainer._training_step(batch, amp_enabled=False)
        assert torch.isfinite(loss).item()
        assert loss.item() > 0
        assert token_count == labels.numel()


def test_gradients_nonzero_and_optimizer_step_changes():
    """Require finite loss and finite, non-zero gradients; prove optimizer steps change weights."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg = _setup_tiny_env(tmp_dir)
        pretrainer = GirivinityPretrainer(cfg)
        pretrainer.load_data()
        optimizer = torch.optim.AdamW(pretrainer.model.parameters(), lr=cfg.learning_rate)

        batch = next(iter(pretrainer.train_loader))
        loss, _ = pretrainer._training_step(batch, amp_enabled=False)
        loss.backward()

        # Require finite and non-zero gradients on trainable parameters
        grads = [p.grad for p in pretrainer.model.parameters() if p.requires_grad and p.grad is not None]
        assert len(grads) > 0, "No gradients populated!"
        for g in grads:
            assert torch.all(torch.isfinite(g)), "Found non-finite gradient!"
        assert any(torch.any(g != 0) for g in grads), "All gradients were zero!"

        # Track weights before and after optimizer step
        params_before = [p.clone().detach() for p in pretrainer.model.parameters() if p.requires_grad]
        optimizer.step()
        params_after = [p.clone().detach() for p in pretrainer.model.parameters() if p.requires_grad]

        changed = [not torch.equal(b, a) for b, a in zip(params_before, params_after)]
        assert any(changed), "Optimizer step did not update any model parameters!"


def test_gradient_accumulation_parity():
    """Compare gradient accumulation with equivalent larger batch within documented tolerance."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg = _setup_tiny_env(tmp_dir)

        # Fix RNG seed for exact initialization equivalence
        torch.manual_seed(42)
        model_accum = GirivinityModel(cfg.model_config)

        torch.manual_seed(42)
        model_single = GirivinityModel(cfg.model_config)

        # Batch 1 and Batch 2
        batch1 = {
            "input_ids": torch.randint(0, 128, (2, 16)),
            "labels": torch.randint(0, 128, (2, 16)),
        }
        batch2 = {
            "input_ids": torch.randint(0, 128, (2, 16)),
            "labels": torch.randint(0, 128, (2, 16)),
        }
        combined_batch = {
            "input_ids": torch.cat([batch1["input_ids"], batch2["input_ids"]], dim=0),
            "labels": torch.cat([batch1["labels"], batch2["labels"]], dim=0),
        }

        # 1. Gradient accumulation: 2 mini-batches of size 2, accumulated over 2 steps
        opt_accum = torch.optim.SGD(model_accum.parameters(), lr=1e-2)
        opt_accum.zero_grad()

        # Forward + backward 1
        logits1, _ = model_accum(batch1["input_ids"])
        loss1 = F.cross_entropy(logits1.reshape(-1, logits1.size(-1)), batch1["labels"].reshape(-1))
        (loss1 / 2.0).backward()

        # Forward + backward 2
        logits2, _ = model_accum(batch2["input_ids"])
        loss2 = F.cross_entropy(logits2.reshape(-1, logits2.size(-1)), batch2["labels"].reshape(-1))
        (loss2 / 2.0).backward()

        opt_accum.step()

        # 2. Single larger batch: 1 batch of size 4
        opt_single = torch.optim.SGD(model_single.parameters(), lr=1e-2)
        opt_single.zero_grad()

        logits_comb, _ = model_single(combined_batch["input_ids"])
        loss_comb = F.cross_entropy(logits_comb.reshape(-1, logits_comb.size(-1)), combined_batch["labels"].reshape(-1))
        loss_comb.backward()

        opt_single.step()

        # Compare updated weights: mean reduction cross-entropy over 2 batches vs 1 combined batch
        for (name, p_acc), (_, p_sng) in zip(model_accum.named_parameters(), model_single.named_parameters()):
            assert torch.allclose(p_acc, p_sng, atol=1e-6, rtol=1e-6), f"Mismatch in {name} after accumulation!"


def test_checkpoint_save_and_reload_consistency():
    """Test checkpoint save/load and compare outputs before and after reload."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg = _setup_tiny_env(tmp_dir)
        pretrainer = GirivinityPretrainer(cfg)
        pretrainer.load_data()

        # Run 1 optimizer step
        batch = next(iter(pretrainer.train_loader))
        optimizer = torch.optim.AdamW(pretrainer.model.parameters(), lr=cfg.learning_rate)
        loss, _ = pretrainer._training_step(batch, amp_enabled=False)
        loss.backward()
        optimizer.step()

        # Compute output before checkpointing
        test_inputs = torch.randint(0, 128, (2, 8))
        pretrainer.model.eval()
        with torch.no_grad():
            logits_before, _ = pretrainer.model(test_inputs)

        # Save checkpoint via pretrainer
        ckpt_dir = Path(cfg.output_dir) / "step-1"
        pretrainer._save_checkpoint(step=1)
        assert (ckpt_dir / "pytorch_model.bin").exists()
        assert (ckpt_dir / "config.json").exists()

        # Reload model using load_pretrained
        reloaded_model = GirivinityModel.load_pretrained(str(ckpt_dir))
        reloaded_model.eval()
        with torch.no_grad():
            logits_after, _ = reloaded_model(test_inputs)

        # Output logits must match exactly
        assert torch.allclose(logits_before, logits_after, atol=1e-6), "Logits differed after checkpoint reload!"
