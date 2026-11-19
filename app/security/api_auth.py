from __future__ import annotations

import hmac
import os
import re
from enum import Enum
from pathlib import Path
from typing import Any


class RouteAccessLevel(str, Enum):
    PUBLIC = "public"
    AUTHENTICATED = "authenticated"
    ADMIN = "admin"
    INTERNAL = "internal"


# Canonical Route Classification Inventory according to Section 5.7
ROUTE_INVENTORY: dict[str, dict[str, Any]] = {
    # Public routes
    "/health": {"method": "GET", "level": RouteAccessLevel.PUBLIC, "description": "Liveness probe"},
    "/health/deep": {"method": "GET", "level": RouteAccessLevel.PUBLIC, "description": "Deep health check"},
    "/metrics": {"method": "GET", "level": RouteAccessLevel.PUBLIC, "description": "Prometheus metrics"},
    "/modules": {"method": "GET", "level": RouteAccessLevel.PUBLIC, "description": "Registered module listing"},
    "/v1/system/status": {"method": "GET", "level": RouteAccessLevel.PUBLIC, "description": "System status"},
    "/v1/system/metrics": {"method": "GET", "level": RouteAccessLevel.PUBLIC, "description": "V1 metrics text"},
    "/v1/cve/{cve_id}": {"method": "GET", "level": RouteAccessLevel.PUBLIC, "description": "Public CVE query"},
    "/v1/cve/search": {"method": "POST", "level": RouteAccessLevel.PUBLIC, "description": "Public CVE search"},
    "/v1/mitre/{technique_id}": {"method": "GET", "level": RouteAccessLevel.PUBLIC, "description": "Public ATT&CK query"},
    "/v1/ioc/check": {"method": "POST", "level": RouteAccessLevel.PUBLIC, "description": "Public IOC check"},
    "/v1/chat": {"method": "POST", "level": RouteAccessLevel.PUBLIC, "description": "Public chat with rate limits"},
    "/chat/message": {"method": "POST", "level": RouteAccessLevel.PUBLIC, "description": "Public chat message endpoint"},
    "/v1/analyze/code": {"method": "POST", "level": RouteAccessLevel.PUBLIC, "description": "Static code analysis"},
    "/v1/analyze/threat": {"method": "POST", "level": RouteAccessLevel.PUBLIC, "description": "Threat analysis"},

    # Authenticated User / Tenant routes
    "/v1/feedback": {"method": "POST", "level": RouteAccessLevel.AUTHENTICATED, "description": "User feedback"},
    "/v1/agents/{agent_type_id}/run": {"method": "POST", "level": RouteAccessLevel.AUTHENTICATED, "description": "Run adaptive agent"},
    "/v1/agents/{agent_type_id}/feedback": {"method": "POST", "level": RouteAccessLevel.AUTHENTICATED, "description": "Agent feedback"},
    "/agents/run": {"method": "POST", "level": RouteAccessLevel.AUTHENTICATED, "description": "Run agent orchestrator"},
    "/tenant/security/config": {"method": "GET/PUT", "level": RouteAccessLevel.AUTHENTICATED, "description": "Tenant security config"},
    "/skills/feedback": {"method": "POST", "level": RouteAccessLevel.AUTHENTICATED, "description": "Submit skill feedback"},
    "/admin/feedback": {"method": "POST", "level": RouteAccessLevel.AUTHENTICATED, "description": "Submit user feedback"},

    # Admin Only routes
    "/admin/notifications": {"method": "GET", "level": RouteAccessLevel.ADMIN, "description": "Successor notifications"},
    "/admin/approve-successor/{version}": {"method": "POST", "level": RouteAccessLevel.ADMIN, "description": "Approve successor model"},
    "/admin/reject-successor/{version}": {"method": "POST", "level": RouteAccessLevel.ADMIN, "description": "Reject successor model"},
    "/admin/model-versions": {"method": "GET", "level": RouteAccessLevel.ADMIN, "description": "List internal model versions"},
    "/v1/training/trigger": {"method": "POST", "level": RouteAccessLevel.ADMIN, "description": "Trigger self-training loop"},
    "/v1/training/status": {"method": "GET", "level": RouteAccessLevel.ADMIN, "description": "Training engine status"},
    "/v1/agents": {"method": "POST", "level": RouteAccessLevel.ADMIN, "description": "Register custom agent"},
    "/security/mode/{mode}": {"method": "POST", "level": RouteAccessLevel.ADMIN, "description": "Change system security mode"},
    "/security/emergency/resolve": {"method": "POST", "level": RouteAccessLevel.ADMIN, "description": "Resolve emergency mode"},
    "/security/events": {"method": "GET", "level": RouteAccessLevel.ADMIN, "description": "Audit security event logs"},

    # Internal / Protected routes
    "/security/rasp/integrity-check": {"method": "GET", "level": RouteAccessLevel.INTERNAL, "description": "File integrity checks"},
    "/cuda/generate": {"method": "POST", "level": RouteAccessLevel.INTERNAL, "description": "CUDA kernel generation"},
    "/cuda/benchmark": {"method": "POST", "level": RouteAccessLevel.INTERNAL, "description": "CUDA kernel benchmarking"},
}


class SecurityAuthError(Exception):
    def __init__(self, message: str, status_code: int = 401) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def sanitize_path(base_dir: str | Path, subpath: str) -> Path:
    """
    Prevent path traversal vulnerabilities.
    Verifies that the target path resolves strictly within base_dir.
    Raises SecurityAuthError if path traversal is detected.
    """
    if "\x00" in subpath:
        raise SecurityAuthError("Invalid null byte in path", status_code=400)

    # Disallow backslashes and relative path separators in path parameters
    if ".." in subpath or subpath.startswith("/") or subpath.startswith("\\"):
        raise SecurityAuthError(f"Path traversal detected in '{subpath}'", status_code=400)

    base = Path(base_dir).resolve()
    target = (base / subpath).resolve()

    try:
        target.relative_to(base)
    except ValueError as exc:
        raise SecurityAuthError(f"Path traversal detected outside base directory: {subpath}", status_code=400) from exc

    return target


def verify_admin_token(token: str | None) -> bool:
    """Constant-time token verification for admin routes."""
    admin_key = os.environ.get("ADMIN_API_KEY", "").strip()
    if not admin_key:
        # If no admin key configured, fail closed
        return False
    if not token:
        return False
    return hmac.compare_digest(token.strip(), admin_key)


def verify_tenant_key(key: str | None) -> bool:
    """Verifies that a non-empty tenant API key is supplied."""
    if not key or not key.strip():
        return False
    if key.strip() == "anonymous":
        return False
    return True


def check_request_authorization(
    path: str,
    headers: dict[str, str],
) -> tuple[bool, str]:
    """
    Enforces route access control according to the ROUTE_INVENTORY classification.
    Returns (authorized: bool, reason: str).
    """
    # Normalize path
    norm_path = path.rstrip("/")
    if not norm_path:
        norm_path = "/"

    # Find matching classification (handle parameterized routes like /admin/approve-successor/v1)
    matched_entry = None
    for pattern, meta in ROUTE_INVENTORY.items():
        # Match pattern
        regex_pattern = "^" + re.sub(r"\{[a-zA-Z0-9_]+\}", r"[a-zA-Z0-9_.-]+", pattern) + "$"
        if re.match(regex_pattern, norm_path):
            matched_entry = meta
            break

    if not matched_entry:
        # Default policy: if route unknown, treat as requiring authentication
        matched_entry = {"level": RouteAccessLevel.AUTHENTICATED}

    level = matched_entry["level"]

    if level == RouteAccessLevel.PUBLIC:
        return True, "Public route"

    # Extract credentials
    auth_header = headers.get("authorization", "")
    bearer_token = auth_header[7:].strip() if auth_header.startswith("Bearer ") else ""
    x_api_key = headers.get("x-api-key", "").strip() or bearer_token

    if level == RouteAccessLevel.ADMIN:
        admin_key = headers.get("x-admin-key", "").strip() or bearer_token or x_api_key
        if verify_admin_token(admin_key):
            return True, "Admin authorized"
        return False, "Admin authorization required (invalid or missing admin credentials)"

    if level == RouteAccessLevel.INTERNAL:
        internal_key = headers.get("x-internal-key", "").strip() or headers.get("x-admin-key", "").strip()
        admin_key = os.environ.get("ADMIN_API_KEY", "").strip()
        if admin_key and internal_key and hmac.compare_digest(internal_key, admin_key):
            return True, "Internal authorized"
        return False, "Internal access only"

    if level == RouteAccessLevel.AUTHENTICATED:
        if verify_tenant_key(x_api_key):
            return True, "Tenant authorized"
        return False, "Authentication required (missing or invalid X-API-Key)"

    return False, "Unauthorized"
