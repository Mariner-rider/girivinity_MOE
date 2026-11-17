import pytest
import torch
import yaml
from pathlib import Path

from model.architecture import GirivinityConfig, GirivinityModel, MoELayer
from app.training.pretrain import PretrainConfig


def test_config_yaml_3b_moe_and_queue_db():
    cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    arch = cfg["model"]["architecture"]
    assert arch["n_routed_experts"] == 16, "Expected Option B 16 routed experts"
    assert arch["n_activated_experts"] == 4, "Expected Option B 4 activated experts"
    assert arch["n_shared_experts"] == 2, "Expected Option B 2 shared experts"
    assert arch["moe_intermediate_size"] == 1024, "Expected Option B 1024 intermediate size"
    assert cfg["training"]["queue_db"] == "data/training.db"


def test_pretrain_config_train_seq_len():
    pt_cfg = PretrainConfig.from_yaml("config.yaml")
    assert pt_cfg.train_seq_len == 4096


def test_moe_gradient_checkpointing_avoids_recompute_mutation():
    cfg = GirivinityConfig(
        dim=64,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=200,
        max_seq_len=32,
        n_routed_experts=8,
        n_activated_experts=2,
        n_shared_experts=1,
        moe_intermediate_size=32,
        moe_start_layer=1,
    )
    model = GirivinityModel(cfg)
    model.train()
    model.gradient_checkpointing_enable()

    moe_layer = model.layers[1].ffn
    assert moe_layer.defer_bias_update is True
    b_init = moe_layer.routing_bias.clone()

    ids = torch.randint(0, cfg.vocab_size, (2, 10))
    out, _ = model(ids)
    assert torch.allclose(moe_layer.routing_bias, b_init), "Routing bias should not mutate in forward under checkpointing"

    out.sum().backward()
    assert torch.allclose(moe_layer.routing_bias, b_init), "Routing bias should not mutate in backward recomputation"

    model.update_moe_biases()
    assert not torch.allclose(moe_layer.routing_bias, b_init), "Routing bias should update when update_moe_biases() is called"
