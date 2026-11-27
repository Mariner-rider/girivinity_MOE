import json as _json
import logging
import time as _time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import Body, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.api.routes.admin import router as admin_router
from app.api.routes.chat import router as chat_router
from app.api.routes.cuda import router as cuda_router
from app.api.routes.health import router as health_router
from app.api.routes.skills import router as skills_router
from app.core.config import get_settings
from app.core.self_trainer import SelfTrainer
from app.core.successor_engine import SuccessorEngine
from app.core.system_config import REQUIRED_MODULES, get_system_config
from app.monitoring.logging import configure_logging
from app.monitoring.metrics import REQUEST_COUNTER
from app.security.rasp import RASPEngine
from app.security.rasp_api import rasp_router
from llm_loader import GirivinityLoader

try:
    from app.engines.user_behavior_engine import FeedbackHarvester
except (ImportError, RuntimeError, OSError):
    FeedbackHarvester = None  # type: ignore

logger = logging.getLogger(__name__)

_llm_runtime: dict[str, Any] = {}
_self_trainer_process: Any | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level, structured=settings.structured_logging)
    _llm_runtime["config"] = get_system_config()
    app.state.rasp = RASPEngine(config={"security": getattr(settings, "security", {})})
    app.state.rasp_engine = app.state.rasp
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=False)
    except RuntimeError:
        pass  # already set (spawn re-imports the app)
    global _self_trainer_process
    _self_trainer_process = SelfTrainer.start()   # starts if enabled
    SuccessorEngine.start()                        # starts if enabled
    from app.core.cuda_crawler import CUDACrawler
    CUDACrawler().bootstrap_async()
    if settings.auto_load_model:
        _llm_runtime["runtime"] = GirivinityLoader().get_model()
    yield
    # Clean shutdown of workers and runtime
    try:
        SelfTrainer.stop()
    except (RuntimeError, OSError, ValueError, AttributeError) as exc:
        logger.warning("Error stopping SelfTrainer: %s", exc)
    try:
        SuccessorEngine.stop()
    except (RuntimeError, OSError, ValueError, AttributeError) as exc:
        logger.warning("Error stopping SuccessorEngine: %s", exc)
    _llm_runtime.clear()


app = FastAPI(title="Girivinity", lifespan=lifespan)
app.include_router(chat_router)
app.include_router(health_router)
app.include_router(admin_router)
app.include_router(skills_router)
app.include_router(cuda_router)
app.include_router(rasp_router)


def _start_self_trainer_once() -> None:
    global _self_trainer_process
    if _self_trainer_process is None or not _self_trainer_process.is_alive():
        _self_trainer_process = SelfTrainer.start()


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    REQUEST_COUNTER.labels(path=request.url.path, method=request.method).inc()
    return await call_next(request)


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "model_loaded": "runtime" in _llm_runtime,
            "modules": list(REQUIRED_MODULES),
        }
    )


@app.get("/modules")
def modules() -> JSONResponse:
    config = _llm_runtime.get("config") or get_system_config()
    return JSONResponse({"modules": config.get("modules", {})})


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)




# --- v1 production endpoints -------------------------------------------------
def _ndjson_stream(text: str):
    for token in text.split(" "):
        yield _json.dumps({"token": token + " ", "done": False}) + "\n"
    yield _json.dumps({"token": "", "done": True}) + "\n"


@app.post("/v1/chat")
async def v1_chat(
    text: Annotated[str, Form()],
    image: Annotated[UploadFile | None, File()] = None,
    audio: Annotated[UploadFile | None, File()] = None,
    video: Annotated[UploadFile | None, File()] = None,
    user_id: Annotated[str | None, Form()] = None,
):
    from app.cognition.sentiment_engine import SentimentEngine

    sentiment = SentimentEngine().analyze(text)
    crisis = bool(getattr(sentiment, "crisis_signal", False) if not isinstance(sentiment, dict) else sentiment.get("crisis_signal"))
    prefix = "I’m here with you. If you may be in immediate danger, contact local emergency services or a trusted person now. " if crisis else ""
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(_time.time())
    try:
        from agent_controller import AgentController

        controller = AgentController()
        output = controller.execute(text, user_id=user_id) if hasattr(controller, "execute") else str(controller.run(text))
    except (RuntimeError, ValueError, TypeError, KeyError, OSError, AttributeError, ImportError) as exc:
        logger.error("v1 chat failed: %s", exc)
        return {
            "id": chat_id,
            "object": "chat.completion",
            "created": created,
            "model": "girivinity",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "I'm unable to process that request right now. Please try again.",
                    },
                    "finish_reason": "error",
                    "index": 0,
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
    return StreamingResponse(_ndjson_stream(prefix + str(output)), media_type="application/x-ndjson")


@app.post("/v1/analyze/code")
async def v1_analyze_code(payload: Annotated[dict, Body()]):
    code = payload.get("code", "")
    findings = []
    if "eval(" in code or "exec(" in code: findings.append({"type": "dangerous-eval", "severity": "high"})
    if "subprocess" in code and "shell=True" in code: findings.append({"type": "shell-injection", "severity": "critical"})
    severity = "critical" if any(f["severity"] == "critical" for f in findings) else "high" if findings else "low"
    return {"findings": findings, "severity": severity, "mitigations": ["Validate input", "Avoid dynamic execution", "Run static analysis in CI"]}


@app.post("/v1/analyze/threat")
async def v1_analyze_threat(payload: Annotated[dict, Body()]):
    import re
    text = payload.get("text", "")
    cves = re.findall(r"CVE-\d{4}-\d{4,}", text, re.IGNORECASE)
    iocs = re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b|\b[a-fA-F0-9]{32,64}\b", text) if payload.get("include_ioc", True) else []
    techniques = re.findall(r"T\d{4}(?:\.\d{3})?", text)
    risk = min(1.0, 0.2 + 0.2 * len(cves) + 0.1 * len(iocs) + 0.2 * len(techniques))
    return {"cves": cves, "techniques": techniques, "iocs": iocs, "risk_score": risk, "report": f"Detected {len(cves)} CVEs, {len(techniques)} ATT&CK techniques, and {len(iocs)} IOCs."}


@app.get("/v1/cve/{cve_id}")
async def v1_cve(cve_id: str):
    return {"id": cve_id.upper(), "source": "NVD", "status": "lookup_unavailable_offline", "url": f"https://nvd.nist.gov/vuln/detail/{cve_id.upper()}"}


@app.post("/v1/cve/search")
async def v1_cve_search(payload: Annotated[dict, Body()]):
    return {
        "results": [],
        "keyword": payload.get("keyword", ""),
        "note": "Local NVD index not built. Set NVD_API_KEY and run the NVD sync job to enable.",
    }


@app.get("/v1/mitre/{technique_id}")
async def v1_mitre(technique_id: str):
    return {"technique_id": technique_id.upper(), "url": f"https://attack.mitre.org/techniques/{technique_id.upper().replace('.', '/')}/", "name": "ATT&CK technique record"}


@app.post("/v1/ioc/check")
async def v1_ioc_check(payload: Annotated[dict, Body()]):
    import re
    value = payload.get("value", "")
    vt = "hash" if re.fullmatch(r"[a-fA-F0-9]{32,64}", value) else "ip" if re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", value) else "url" if value.startswith(("http://", "https://")) else "domain"
    return {"is_ioc": vt in {"hash", "ip", "url", "domain"}, "value_type": vt, "feeds": ["local_regex", "configured_feeds"]}


@app.get("/v1/system/status")
async def v1_system_status():
    queue_size, last_cycle, rag_docs = 0, None, 0
    try:
        from app.core.db import fetchone
        row = fetchone("SELECT COUNT(*) FROM training_queue WHERE status='pending'")
        queue_size = row[0] if row else 0
    except (ImportError, OSError, RuntimeError, ValueError, TypeError) as exc:
        logger.debug("Training queue status unavailable: %s", exc)
    try:
        log = Path("logs/self_training.jsonl")
        if log.exists():
            lines = [line for line in log.read_text().splitlines() if line.strip()]
            if lines:
                last_cycle = _json.loads(lines[-1]).get("timestamp")
    except (OSError, ValueError, TypeError) as exc:
        logger.debug("Self-training log unavailable: %s", exc)
    try:
        import chromadb
        import yaml as _yaml
        cfg = _yaml.safe_load(Path("config.yaml").read_text())
        col = chromadb.PersistentClient(path=cfg["rag"]["chroma_path"]).get_or_create_collection("girivinity_knowledge")
        rag_docs = col.count()
    except (ImportError, OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        logger.debug("RAG status unavailable: %s", exc)

    param_profile = "unknown"
    try:
        from model.architecture import GirivinityConfig
        cfg = GirivinityConfig.from_yaml()
        counts = cfg.count_parameters()
        total_b = counts["total_parameters"] / 1e9
        active_b = counts["active_parameters"] / 1e9
        param_profile = f"{total_b:.1f}B ({active_b:.1f}B active)"
    except (ImportError, OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        logger.debug("Model parameter profile unavailable: %s", exc)

    return {
        "model_loaded": "runtime" in _llm_runtime,
        "backend": "girivinity",
        "current_generation": 1,
        "parameter_scale": param_profile,
        "rag_docs": rag_docs,
        "training_queue_size": queue_size,
        "last_training_cycle": last_cycle,
    }

@app.get("/v1/system/metrics")
async def v1_system_metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/v1/training/status")
async def v1_training_status():
    return {"self_improvement_loop": {"available": True}, "successor_engine": {"available": True}}


@app.post("/v1/training/trigger")
async def v1_training_trigger(x_admin_key: str | None = Header(default=None, alias="X-Admin-Key")):
    from app.security.api_auth import verify_admin_token
    if not verify_admin_token(x_admin_key):
        raise HTTPException(status_code=401, detail="Unauthorized: Valid X-Admin-Key required")
    from starlette.concurrency import run_in_threadpool

    from app.core.self_trainer import SelfTrainer

    def _run() -> int:
        trainer = SelfTrainer()
        trainer._check_and_train()
        return 0

    await run_in_threadpool(_run)
    return {"training_triggered": True, "finished_at": _time.time()}


@app.post("/v1/feedback")
async def v1_feedback(payload: Annotated[dict, Body()]):
    if FeedbackHarvester is None:
        return {"stored": False, "reason": "FeedbackHarvester unavailable"}
    FeedbackHarvester().collect_explicit(payload["interaction_id"], int(payload["rating"]), payload.get("correction"), payload.get("user_id"))
    return {"stored": True}

# --- adaptive agent platform endpoints --------------------------------------
_adaptive_runtime: dict[str, Any] = {}


def _get_agent_registry():
    if "agent_registry" not in _adaptive_runtime:
        from app.agents.agent_registry import AgentRegistry

        _adaptive_runtime["agent_registry"] = AgentRegistry.from_config()
    return _adaptive_runtime["agent_registry"]


def _get_adaptive_executor():
    if "adaptive_executor" not in _adaptive_runtime:
        from agent_controller import LocalLLMEngine
        from app.agents.adaptive_agent_executor import (
            AdaptiveAgentExecutor,
            BasicToolRegistry,
            InMemoryRAG,
        )
        from app.agents.capability_merger import CapabilityMerger
        from app.security.policy import SecurityGuard

        registry = _get_agent_registry()
        merger = CapabilityMerger({"adaptive_agents": {"merge_threshold": 50}}, registry, None, SecurityGuard())
        _adaptive_runtime["adaptive_executor"] = AdaptiveAgentExecutor(
            LocalLLMEngine(), registry, merger, BasicToolRegistry(), InMemoryRAG()
        )
    return _adaptive_runtime["adaptive_executor"]


@app.get("/v1/agents")
async def v1_agents_list():
    registry = _get_agent_registry()
    return {"agents": [agent.to_dict() for agent in registry.list_all()]}


@app.get("/v1/agents/search")
async def v1_agents_search(q: str = Query(..., min_length=1)):
    registry = _get_agent_registry()
    return {"query": q, "agents": [agent.to_dict() for agent in registry.search(q)]}


@app.post("/v1/agents")
async def v1_agents_register(payload: Annotated[dict, Body()], x_admin_key: str | None = Header(default=None, alias="X-Admin-Key")):
    from app.security.api_auth import verify_admin_token
    if not verify_admin_token(x_admin_key):
        raise HTTPException(status_code=401, detail="Unauthorized: Valid X-Admin-Key required to register agent definitions")
    from app.agents.agent_registry import AgentTypeDefinition

    required = ["agent_type_id", "display_name", "description", "base_system_prompt"]
    missing = [key for key in required if not payload.get(key)]
    if missing:
        raise HTTPException(status_code=422, detail={"missing": missing})
    registry = _get_agent_registry()
    agent_def = AgentTypeDefinition(
        agent_type_id=str(payload["agent_type_id"]).strip().lower().replace(" ", "_"),
        display_name=str(payload["display_name"]),
        description=str(payload["description"]),
        base_system_prompt=str(payload["base_system_prompt"]),
        available_tools=[str(tool) for tool in payload.get("available_tools", [])],
        adapter_path=str(payload.get("adapter_path") or f"models/adapters/{payload['agent_type_id']}/latest"),
        capability_version=int(payload.get("capability_version", 1)),
        user_count=int(payload.get("user_count", 0)),
        tags=[str(tag).lower() for tag in payload.get("tags", [])],
    )
    registry.register(agent_def)
    return {"registered": True, "agent": agent_def.to_dict()}


@app.get("/v1/agents/{agent_type_id}")
async def v1_agents_get(agent_type_id: str):
    registry = _get_agent_registry()
    agent = registry.get(agent_type_id)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Unknown agent type: {agent_type_id}")
    return agent.to_dict()


@app.post("/v1/agents/{agent_type_id}/run")
async def v1_agents_run(agent_type_id: str, payload: Annotated[dict, Body()]):
    user_id = payload.get("user_id")
    task = payload.get("task")
    if not user_id or not task:
        raise HTTPException(status_code=422, detail="user_id and task are required")
    executor = _get_adaptive_executor()
    try:
        return await executor.execute(agent_type_id, str(user_id), str(task), payload.get("user_context"))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/v1/agents/{agent_type_id}/stats")
async def v1_agents_stats(agent_type_id: str):
    registry = _get_agent_registry()
    try:
        return registry.stats(agent_type_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/v1/agents/{agent_type_id}/feedback")
async def v1_agents_feedback(agent_type_id: str, payload: Annotated[dict, Body()]):
    registry = _get_agent_registry()
    if registry.get(agent_type_id) is None:
        raise HTTPException(status_code=404, detail=f"Unknown agent type: {agent_type_id}")
    user_id = str(payload.get("user_id") or "")
    interaction_id = str(payload.get("interaction_id") or "")
    rating = int(payload.get("rating", 0))
    correction = str(payload.get("correction") or "")
    if not user_id or not interaction_id or rating < 1 or rating > 5:
        raise HTTPException(status_code=422, detail="user_id, interaction_id, and rating 1-5 are required")
    example = {
        "instruction": f"Improve response for interaction {interaction_id}",
        "input": "",
        "output": correction or f"User rated interaction {interaction_id} as {rating}/5.",
        "quality_score": rating / 5.0,
    }
    delta_id = registry.submit_capability_delta(
        user_id=user_id,
        agent_type_id=agent_type_id,
        delta_description=f"Feedback for interaction {interaction_id}: rating={rating}",
        training_example=example,
        quality_score=rating / 5.0,
    )
    if FeedbackHarvester is not None:
        FeedbackHarvester().collect_explicit(interaction_id, rating, correction or None, user_id)
    return {"stored": True, "delta_id": delta_id, "agent_type": agent_type_id, "rating": rating}
