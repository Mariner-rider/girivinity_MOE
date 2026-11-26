"""Centralized API authentication, authorization, rate limiting, and audit logging."""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from fastapi import Depends, Header, HTTPException, Request, status

logger = logging.getLogger("app.security.api_auth")
audit_logger = logging.getLogger("app.security.audit")


class AccessTier(str, Enum):
    PUBLIC_LIVENESS = "public_liveness"
    PUBLIC_CHAT = "public_chat"
    AUTHENTICATED_USER = "authenticated_user"
    SERVICE_INTERNAL = "service_internal"
    ADMIN_ONLY = "admin_only"


class Role(str, Enum):
    USER = "user"
    SERVICE = "service"
    ADMIN = "admin"


@dataclass(frozen=True)
class Principal:
    principal_id: str
    role: Role
    auth_method: str


# Explicit classification of every live route on `app` as `(METHOD, PATH) -> AccessTier`.
# Any newly added route not present in this map will fail the route-inventory regression test.
ROUTE_CLASSIFICATION: dict[tuple[str, str], AccessTier] = {
    # Public liveness & OpenAPI schema
    ("GET", "/health"): AccessTier.PUBLIC_LIVENESS,
    ("GET", "/openapi.json"): AccessTier.PUBLIC_LIVENESS,
    ("GET", "/docs"): AccessTier.PUBLIC_LIVENESS,
    ("GET", "/docs/oauth2-redirect"): AccessTier.PUBLIC_LIVENESS,
    ("GET", "/redoc"): AccessTier.PUBLIC_LIVENESS,
    # Public rate-limited chat
    ("POST", "/chat/message"): AccessTier.PUBLIC_CHAT,
    ("POST", "/chat/message/stream"): AccessTier.PUBLIC_CHAT,
    ("POST", "/v1/chat"): AccessTier.PUBLIC_CHAT,
    # Service / internal observability & inspection
    ("GET", "/health/deep"): AccessTier.SERVICE_INTERNAL,
    ("GET", "/modules"): AccessTier.SERVICE_INTERNAL,
    ("GET", "/metrics"): AccessTier.SERVICE_INTERNAL,
    ("GET", "/v1/system/status"): AccessTier.SERVICE_INTERNAL,
    ("GET", "/v1/system/metrics"): AccessTier.SERVICE_INTERNAL,
    ("GET", "/v1/training/status"): AccessTier.SERVICE_INTERNAL,
    ("POST", "/rasp/v1/inspect/input"): AccessTier.SERVICE_INTERNAL,
    ("POST", "/rasp/v1/inspect/output"): AccessTier.SERVICE_INTERNAL,
    ("GET", "/rasp/v1/threats/summary"): AccessTier.SERVICE_INTERNAL,
    # Authenticated user endpoints
    ("POST", "/v1/feedback"): AccessTier.AUTHENTICATED_USER,
    ("POST", "/v1/analyze/code"): AccessTier.AUTHENTICATED_USER,
    ("POST", "/v1/analyze/threat"): AccessTier.AUTHENTICATED_USER,
    ("GET", "/v1/cve/{cve_id}"): AccessTier.AUTHENTICATED_USER,
    ("POST", "/v1/cve/search"): AccessTier.AUTHENTICATED_USER,
    ("GET", "/v1/mitre/{technique_id}"): AccessTier.AUTHENTICATED_USER,
    ("POST", "/v1/ioc/check"): AccessTier.AUTHENTICATED_USER,
    ("GET", "/v1/agents"): AccessTier.AUTHENTICATED_USER,
    ("GET", "/v1/agents/search"): AccessTier.AUTHENTICATED_USER,
    ("GET", "/v1/agents/{agent_type_id}"): AccessTier.AUTHENTICATED_USER,
    ("POST", "/v1/agents/{agent_type_id}/run"): AccessTier.AUTHENTICATED_USER,
    ("GET", "/v1/agents/{agent_type_id}/stats"): AccessTier.AUTHENTICATED_USER,
    ("GET", "/skills/"): AccessTier.AUTHENTICATED_USER,
    ("GET", "/skills/{slug}"): AccessTier.AUTHENTICATED_USER,
    ("POST", "/skills/feedback"): AccessTier.AUTHENTICATED_USER,
    ("POST", "/cuda/generate"): AccessTier.AUTHENTICATED_USER,
    ("POST", "/cuda/benchmark"): AccessTier.AUTHENTICATED_USER,
    ("GET", "/cuda/types"): AccessTier.AUTHENTICATED_USER,
    # Admin-only endpoints
    ("GET", "/admin/notifications"): AccessTier.ADMIN_ONLY,
    ("POST", "/admin/approve-successor/{version}"): AccessTier.ADMIN_ONLY,
    ("POST", "/admin/reject-successor/{version}"): AccessTier.ADMIN_ONLY,
    ("GET", "/admin/model-versions"): AccessTier.ADMIN_ONLY,
    ("POST", "/admin/feedback"): AccessTier.ADMIN_ONLY,
    ("POST", "/v1/training/trigger"): AccessTier.ADMIN_ONLY,
    ("POST", "/v1/agents"): AccessTier.ADMIN_ONLY,
    ("POST", "/v1/agents/{agent_type_id}/feedback"): AccessTier.ADMIN_ONLY,
    ("POST", "/skills/{slug}/evaluate"): AccessTier.ADMIN_ONLY,
    ("DELETE", "/skills/{slug}"): AccessTier.ADMIN_ONLY,
    ("POST", "/cuda/bootstrap"): AccessTier.ADMIN_ONLY,
    ("POST", "/rasp/v1/webhooks"): AccessTier.ADMIN_ONLY,
    ("GET", "/rasp/v1/threats/stream"): AccessTier.ADMIN_ONLY,
}

# Unmounted route modules in app/api/routes/ (documented for inventory visibility)
UNMOUNTED_ROUTE_MODULES: tuple[str, ...] = (
    "app/api/routes/agents.py",
    "app/api/routes/rasp.py",
    "app/api/routes/security.py",
    "app/api/routes/tenant_security.py",
)

_ENV_KEY_MAP: tuple[tuple[str, Role], ...] = (
    ("GIRIVINITY_ADMIN_API_KEYS", Role.ADMIN),
    ("GIRIVINITY_SERVICE_API_KEYS", Role.SERVICE),
    ("RASP_API_KEYS", Role.SERVICE),
    ("GIRIVINITY_USER_API_KEYS", Role.USER),
)

_FORBIDDEN_PLACEHOLDER_KEYS = frozenset({"default", "changeme", "secret", "test", "admin", "password"})


def _parse_key_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _fingerprint_secret(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()[:12]


def _constant_time_in(candidate: str, allowed: list[str]) -> bool:
    matched = False
    for item in allowed:
        if hmac.compare_digest(candidate.encode("utf-8"), item.encode("utf-8")):
            matched = True
    return matched


def _is_revoked(candidate: str) -> bool:
    revoked_keys = _parse_key_list(os.environ.get("GIRIVINITY_REVOKED_KEYS", ""))
    if revoked_keys and _constant_time_in(candidate, revoked_keys):
        return True
    revoked_hashes = [h.lower() for h in _parse_key_list(os.environ.get("GIRIVINITY_REVOKED_KEY_HASHES", ""))]
    if revoked_hashes:
        digest = hashlib.sha256(candidate.encode("utf-8")).hexdigest().lower()
        if _constant_time_in(digest, revoked_hashes):
            return True
    return False


def _extract_credential(
    authorization: str | None,
    x_api_key: str | None,
    x_rasp_key: str | None,
) -> tuple[str, str] | None:
    if authorization is not None:
        parts = authorization.strip().split(" ", 1)
        if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1].strip():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Malformed Authorization header",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return parts[1].strip(), "bearer"
    if x_api_key is not None:
        if not x_api_key.strip():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Empty X-API-Key header",
            )
        return x_api_key.strip(), "x-api-key"
    if x_rasp_key is not None:
        if not x_rasp_key.strip():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Empty X-RASP-Key header",
            )
        return x_rasp_key.strip(), "x-rasp-key"
    return None


def _authenticate_credential(token: str, auth_method: str) -> Principal | None:
    if not token or token.lower() in _FORBIDDEN_PLACEHOLDER_KEYS:
        return None
    if _is_revoked(token):
        return None

    for env_var, role in _ENV_KEY_MAP:
        configured_keys = [
            k for k in _parse_key_list(os.environ.get(env_var, ""))
            if k.lower() not in _FORBIDDEN_PLACEHOLDER_KEYS
        ]
        if configured_keys and _constant_time_in(token, configured_keys):
            return Principal(
                principal_id=f"{role.value}:{_fingerprint_secret(token)}",
                role=role,
                auth_method=auth_method,
            )

    if auth_method == "bearer" and os.environ.get("GIRIVINITY_ENABLE_SESSION_AUTH", "").lower() == "true":
        try:
            from app.security.session_manager import SessionManager

            session = SessionManager().validate_session(token)
            if session and session.get("user_id"):
                return Principal(
                    principal_id=f"session:{session['user_id']}",
                    role=Role.USER,
                    auth_method="session_bearer",
                )
        except Exception as exc:
            logger.warning("Session validation error: %s", exc)

    return None


def get_correlation_id(request: Request) -> str:
    corr_id = getattr(request.state, "correlation_id", None)
    if not corr_id:
        corr_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.correlation_id = corr_id
    return corr_id


def log_audit_event(
    *,
    principal: Principal | None,
    action: str,
    target: str,
    result: str,
    correlation_id: str,
) -> None:
    """Record a security/mutation audit event without logging secrets or payloads."""
    audit_logger.info(
        "audit principal=%s role=%s action=%s target=%s result=%s correlation_id=%s",
        principal.principal_id if principal else "anonymous",
        principal.role.value if principal else "none",
        action,
        target,
        result,
        correlation_id,
    )


def require_roles(
    *allowed_roles: Role,
    action: str = "api_access",
    audit_mutating: bool = False,
) -> Callable[..., Principal]:
    """FastAPI dependency enforcing authentication, role authorization, and audit logging."""
    allowed_set = frozenset(allowed_roles)

    async def _dependency(
        request: Request,
        authorization: str | None = Header(default=None, alias="Authorization"),
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
        x_rasp_key: str | None = Header(default=None, alias="X-RASP-Key"),
    ) -> Principal:
        corr_id = get_correlation_id(request)
        target = request.url.path

        extracted = _extract_credential(authorization, x_api_key, x_rasp_key)
        if extracted is None:
            log_audit_event(
                principal=None,
                action=action,
                target=target,
                result="denied_missing_credentials",
                correlation_id=corr_id,
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )

        token, auth_method = extracted
        principal = _authenticate_credential(token, auth_method)
        if principal is None:
            log_audit_event(
                principal=None,
                action=action,
                target=target,
                result="denied_invalid_credentials",
                correlation_id=corr_id,
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or revoked credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

        if principal.role not in allowed_set:
            log_audit_event(
                principal=principal,
                action=action,
                target=target,
                result="denied_forbidden_role",
                correlation_id=corr_id,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions for this operation",
            )

        if audit_mutating:
            log_audit_event(
                principal=principal,
                action=action,
                target=target,
                result="authorized",
                correlation_id=corr_id,
            )
        request.state.principal = principal
        return principal

    return _dependency


# Reusable role dependencies
require_authenticated_user = require_roles(Role.USER, Role.SERVICE, Role.ADMIN, action="user_access")
require_service_or_admin = require_roles(Role.SERVICE, Role.ADMIN, action="service_access")
require_admin = require_roles(Role.ADMIN, action="admin_access", audit_mutating=True)


# --- Public chat rate limiter (in-memory fallback + RateLimiter hook) --------
_chat_rate_lock = threading.Lock()
_chat_rate_buckets: dict[str, list[float]] = {}


def enforce_public_chat_rate_limit(request: Request) -> None:
    """Rate-limit public chat endpoints per client IP."""
    client_ip = request.client.host if request.client else "unknown"
    limit = int(os.environ.get("GIRIVINITY_PUBLIC_CHAT_RPM", "50"))
    window_seconds = 60
    now = time.monotonic()

    with _chat_rate_lock:
        timestamps = [t for t in _chat_rate_buckets.get(client_ip, []) if now - t < window_seconds]
        if len(timestamps) >= limit:
            _chat_rate_buckets[client_ip] = timestamps
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Public chat rate limit exceeded. Please retry later.",
                headers={"Retry-After": str(window_seconds)},
            )
        timestamps.append(now)
        _chat_rate_buckets[client_ip] = timestamps


def reset_public_chat_rate_limits() -> None:
    """Clear in-memory rate-limit buckets (intended for unit tests)."""
    with _chat_rate_lock:
        _chat_rate_buckets.clear()
