import pytest
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock

from model.architecture import GirivinityConfig, GirivinityModel
from app.core.self_trainer import SelfTrainer
from app.core.successor_engine import SuccessorEngine


def test_parameter_count_formula_matches_actual_tiny():
    """Verify that the allocation-free count_parameters() formula matches actual tensor instantiation on a tiny model."""
    tiny_cfg = GirivinityConfig(
        dim=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=256,
        max_seq_len=64,
        moe_enabled=True,
        n_routed_experts=4,
        n_activated_experts=2,
        n_shared_experts=1,
        moe_intermediate_size=32,
        moe_start_layer=1,
    )
    
    formula_counts = tiny_cfg.count_parameters()
    
    # Instantiate actual model and count parameters
    model = GirivinityModel(tiny_cfg)
    actual_total = sum(p.numel() for p in model.parameters())
    
    # Note: tied weights (lm_head.weight is embed.weight)
    # in model.parameters(), embed.weight is only visited once if counted properly.
    # Check that our formula matches the unique parameter count.
    assert formula_counts["total_parameters"] == actual_total
    assert formula_counts["active_parameters"] < formula_counts["total_parameters"]


def test_active_yaml_parameter_profile():
    """Check active YAML architecture parameter count without instantiating the full model in memory."""
    cfg = GirivinityConfig.from_yaml("config.yaml")
    counts = cfg.count_parameters()
    
    total = counts["total_parameters"]
    active = counts["active_parameters"]
    
    # Option B MoE is ~5.8B total, ~2.7B active
    assert 5e9 < total < 7e9, f"Unexpected total parameter count: {total}"
    assert 2e9 < active < 4e9, f"Unexpected active parameter count: {active}"


def test_self_trainer_lifecycle_flags():
    """Verify SelfTrainer respects feature flags and idempotent start/stop."""
    mock_cfg = {
        "feature_flags": {"enable_training": False},
        "modules": {"training": {"enabled": True}},
        "training": {"queue_db": "data/training.db"},
        "modules": {"self_training": {
            "interval_seconds": 1800,
            "chunk_threshold": 50,
            "base_model_path": "models/base",
            "adapters_dir": "models/adapters",
            "training_queue_dir": "data/training_queue",
            "event_log_path": "logs/self_training.jsonl",
            "alerts_log_path": "logs/alerts.jsonl",
            "epochs": 3,
            "learning_rate": 0.0002,
            "loss_abort_threshold": 2.0
        }}
    }
    
    with patch("yaml.safe_load", return_value=mock_cfg):
        res = SelfTrainer.start()
        assert res is None, "SelfTrainer should not start when enable_training is False"


def test_successor_engine_lifecycle_flags():
    """Verify SuccessorEngine respects feature flags and idempotent start/stop."""
    mock_cfg = {
        "feature_flags": {"enable_successor_engine": False},
        "successor_engine": {},
        "training": {"queue_db": "data/training.db"}
    }
    
    with patch("yaml.safe_load", return_value=mock_cfg):
        res = SuccessorEngine.start()
        assert res is None, "SuccessorEngine should not start when enable_successor_engine is False"


def test_worker_shutdown_cleans_up():
    """Test SelfTrainer.stop() and SuccessorEngine.stop() cleanup."""
    dummy_proc = MagicMock()
    dummy_proc.is_alive.return_value = True
    
    SelfTrainer._process = dummy_proc
    SelfTrainer.stop()
    dummy_proc.terminate.assert_called_once()
    assert SelfTrainer._process is None

    dummy_proc2 = MagicMock()
    dummy_proc2.is_alive.return_value = True
    SuccessorEngine._process = dummy_proc2
    SuccessorEngine.stop()
    dummy_proc2.terminate.assert_called_once()
    assert SuccessorEngine._process is None
