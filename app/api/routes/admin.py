from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from app.core.successor_engine import SuccessorEngine
from app.security.api_auth import Principal, get_correlation_id, log_audit_event, require_admin

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin)])


def _require_admin(x_admin_key: str | None = Header(default=None, alias="X-Admin-Key")) -> None:
    if not verify_admin_token(x_admin_key):
        raise HTTPException(status_code=401, detail="Unauthorized: Valid X-Admin-Key required")


class FeedbackRequest(BaseModel):
    user_id: str
    score: float


@router.get("/notifications")
async def get_notifications(x_admin_key: str | None = Header(default=None, alias="X-Admin-Key")):
    _require_admin(x_admin_key)
    return {"notifications": SuccessorEngine().get_notifications()}


@router.post("/approve-successor/{version}")
async def approve_successor(version: str, request: Request, principal: Principal = Depends(require_admin)):
    ok = SuccessorEngine().approve_successor(version)
    log_audit_event(
        principal=principal,
        action="admin.approve_successor",
        target=f"/admin/approve-successor/{version}",
        result="approved" if ok else "not_found",
        correlation_id=get_correlation_id(request),
    )
    if not ok:
        raise HTTPException(
            status_code=404, detail=f"Version {version} not found"
        )
    return {"status": "approved", "version": version}


@router.post("/reject-successor/{version}")
async def reject_successor(version: str, request: Request, principal: Principal = Depends(require_admin)):
    SuccessorEngine().reject_successor(version)
    log_audit_event(
        principal=principal,
        action="admin.reject_successor",
        target=f"/admin/reject-successor/{version}",
        result="rejected",
        correlation_id=get_correlation_id(request),
    )
    return {"status": "rejected", "version": version}


@router.get("/model-versions")
async def list_model_versions(x_admin_key: str | None = Header(default=None, alias="X-Admin-Key")):
    _require_admin(x_admin_key)
    from pathlib import Path
    import yaml
    cfg = yaml.safe_load(Path("config.yaml").read_text())
    versions_dir = Path(cfg["successor_engine"]["versions_dir"])
    if not versions_dir.exists():
        return {"versions": []}
    versions = sorted(
        [d.name for d in versions_dir.iterdir() if d.is_dir()],
        reverse=True,
    )
    return {"versions": versions}


@router.post("/feedback")
async def submit_feedback(req: FeedbackRequest, request: Request, principal: Principal = Depends(require_admin)):
    if not 1.0 <= req.score <= 5.0:
        raise HTTPException(
            status_code=400, detail="Score must be between 1.0 and 5.0"
        )
    SuccessorEngine().log_feedback(req.user_id, req.score)
    log_audit_event(
        principal=principal,
        action="admin.submit_feedback",
        target="/admin/feedback",
        result="recorded",
        correlation_id=get_correlation_id(request),
    )
    return {"status": "recorded"}
