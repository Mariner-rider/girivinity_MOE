from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from app.core.cuda_engine import CUDAEngine
from app.security.api_auth import (
    Principal,
    get_correlation_id,
    log_audit_event,
    require_admin,
    require_authenticated_user,
)

router = APIRouter(prefix="/cuda", tags=["cuda"])


class KernelRequest(BaseModel):
    request: str
    hardware_target: str = "auto"
    optimise: bool = True


class BenchmarkRequest(BaseModel):
    kernel_code: str


@router.post("/generate", dependencies=[Depends(require_authenticated_user)])
async def generate_kernel(req: KernelRequest):
    if not req.request.strip():
        raise HTTPException(status_code=400, detail="Request cannot be empty")
    try:
        result = CUDAEngine().generate(request=req.request, hardware_target=req.hardware_target, optimise=req.optimise)
        return {
            "kernel_code": result.kernel_code,
            "kernel_type": result.kernel_type,
            "hardware_target": result.hardware_target,
            "compiled": result.profile.compiled,
            "occupancy_pct": result.profile.occupancy_pct,
            "warp_efficiency_pct": result.profile.warp_efficiency_pct,
            "compile_errors": result.profile.compile_errors,
            "warnings": result.profile.warnings,
            "optimisations_applied": result.optimisations_applied,
            "explanation": result.explanation,
            "improved": result.improved,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/benchmark", dependencies=[Depends(require_authenticated_user)])
async def benchmark_kernel(req: BenchmarkRequest):
    if not req.kernel_code.strip():
        raise HTTPException(status_code=400, detail="Kernel code required")
    profile = CUDAEngine().benchmark(req.kernel_code)
    return {
        "compiled": profile.compiled,
        "occupancy_pct": profile.occupancy_pct,
        "warp_efficiency_pct": profile.warp_efficiency_pct,
        "compile_errors": profile.compile_errors,
        "warnings": profile.warnings,
    }


@router.get("/types", dependencies=[Depends(require_authenticated_user)])
async def list_kernel_types():
    from app.core.cuda_engine import KERNEL_PATTERNS, KERNEL_TEMPLATES

    return {"types": list(KERNEL_PATTERNS.keys()), "templates_available": list(KERNEL_TEMPLATES.keys())}


@router.post("/bootstrap")
async def bootstrap_cuda_knowledge(
    request: Request,
    principal: Principal = Depends(require_admin),
):
    """Trigger CUDA knowledge base crawl in background."""
    from app.core.cuda_crawler import CUDACrawler

    CUDACrawler().bootstrap_async()
    log_audit_event(
        principal=principal,
        action="cuda.bootstrap",
        target="/cuda/bootstrap",
        result="bootstrapping",
        correlation_id=get_correlation_id(request),
    )
    return {"status": "bootstrapping", "sources": len([]), "topics": 12}
