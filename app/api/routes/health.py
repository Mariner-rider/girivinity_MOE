from fastapi import APIRouter, Depends

from app.security.api_auth import require_service_or_admin

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    return {"status": "ok", "service": "girivinity"}


@router.get("/health/deep", dependencies=[Depends(require_service_or_admin)])
async def health_deep():
    """Checks that ChromaDB and embedder are reachable."""
    issues = []
    try:
        from app.core.query_router import get_embedder

        get_embedder()
    except Exception as exc:
        issues.append(f"embedder: {exc}")
    try:
        import chromadb
        from pathlib import Path

        import yaml

        cfg = yaml.safe_load(Path("config.yaml").read_text())
        chromadb.PersistentClient(path=cfg["rag"]["chroma_path"])
    except Exception as exc:
        issues.append(f"chromadb: {exc}")

    try:
        from llm_engine import LLMEngine

        compute = LLMEngine().get_device_info()
    except Exception as exc:
        compute = {
            "device": "cpu",
            "gpu_name": None,
            "gpu_vram_total_gb": None,
            "gpu_vram_free_gb": None,
            "inference_mode": "float32_cpu",
            "error": str(exc),
        }

    model_status: dict[str, object] = {
        "native_model_path": None,
        "checkpoint_present": False,
        "trained_context_verified": False,
    }
    try:
        from pathlib import Path
        import yaml

        cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
        native_path = cfg.get("model", {}).get("native_model_path", "models/base/final")
        ckpt_file = Path(native_path) / "model.pt"
        model_status["native_model_path"] = native_path
        model_status["checkpoint_present"] = ckpt_file.exists()
        arch = cfg.get("model", {}).get("architecture", {})
        model_status["configured_max_seq_len"] = arch.get("max_seq_len")
    except Exception as exc:
        model_status["error"] = str(exc)

    return {
        "status": "ok" if not issues else "degraded",
        "issues": issues,
        "compute": compute,
        "model_status": model_status,
    }
