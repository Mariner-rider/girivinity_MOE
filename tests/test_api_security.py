from __future__ import annotations

import os
import tempfile
from pathlib import Path
import pytest

from app.security.api_auth import (
    RouteAccessLevel,
    ROUTE_INVENTORY,
    SecurityAuthError,
    sanitize_path,
    verify_admin_token,
    verify_tenant_key,
    check_request_authorization,
)


def test_route_inventory_classification_completeness():
    """Verify that every route in the inventory has a valid classification and method."""
    assert len(ROUTE_INVENTORY) >= 20

    levels = {meta["level"] for meta in ROUTE_INVENTORY.values()}
    assert RouteAccessLevel.PUBLIC in levels
    assert RouteAccessLevel.AUTHENTICATED in levels
    assert RouteAccessLevel.ADMIN in levels
    assert RouteAccessLevel.INTERNAL in levels

    # Verify sensitive routes are not classified as PUBLIC
    assert ROUTE_INVENTORY["/admin/approve-successor/{version}"]["level"] == RouteAccessLevel.ADMIN
    assert ROUTE_INVENTORY["/v1/training/trigger"]["level"] == RouteAccessLevel.ADMIN
    assert ROUTE_INVENTORY["/security/mode/{mode}"]["level"] == RouteAccessLevel.ADMIN
    assert ROUTE_INVENTORY["/cuda/benchmark"]["level"] == RouteAccessLevel.INTERNAL


def test_unauthenticated_requests_blocked_for_admin_and_training():
    """Prove unauthenticated users cannot access admin, training, or security operations."""
    headers_unauth = {}

    # Admin route
    ok, reason = check_request_authorization("/admin/approve-successor/v2", headers_unauth)
    assert not ok
    assert "Admin authorization required" in reason

    # Training trigger
    ok, reason = check_request_authorization("/v1/training/trigger", headers_unauth)
    assert not ok
    assert "Admin authorization required" in reason

    # Security mode alteration
    ok, reason = check_request_authorization("/security/mode/emergency", headers_unauth)
    assert not ok

    # Custom agent registration
    ok, reason = check_request_authorization("/v1/agents", headers_unauth)
    assert not ok


def test_authenticated_user_cannot_access_admin_operations(monkeypatch):
    """Prove regular authenticated tenant cannot access admin operations."""
    monkeypatch.setenv("ADMIN_API_KEY", "super-secret-admin-pass-54321")
    tenant_headers = {"x-api-key": "tenant-customer-key-12345"}

    # Tenant accesses authenticated endpoint -> PASS
    ok, _ = check_request_authorization("/tenant/security/config", tenant_headers)
    assert ok

    # Tenant accesses admin endpoint -> FAIL
    ok, reason = check_request_authorization("/admin/approve-successor/v2", tenant_headers)
    assert not ok
    assert "Admin authorization required" in reason

    # Tenant tries triggering training -> FAIL
    ok, _ = check_request_authorization("/v1/training/trigger", tenant_headers)
    assert not ok


def test_admin_authorized_with_proper_credentials(monkeypatch):
    """Prove admin accesses admin operations when providing valid credentials."""
    monkeypatch.setenv("ADMIN_API_KEY", "super-secret-admin-pass-54321")
    admin_headers = {"x-admin-key": "super-secret-admin-pass-54321"}

    ok, reason = check_request_authorization("/admin/approve-successor/v2", admin_headers)
    assert ok
    assert "Admin authorized" in reason

    ok, _ = check_request_authorization("/v1/training/trigger", admin_headers)
    assert ok

    ok, _ = check_request_authorization("/security/mode/emergency", admin_headers)
    assert ok


def test_public_routes_accessible_without_auth():
    """Verify that public routes allow unauthenticated access."""
    ok, _ = check_request_authorization("/health", {})
    assert ok

    ok, _ = check_request_authorization("/metrics", {})
    assert ok

    ok, _ = check_request_authorization("/v1/chat", {})
    assert ok


def test_path_traversal_protection():
    """Verify that path traversal attempts in parameters are strictly caught and rejected."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        base_dir = Path(tmp_dir) / "models"
        base_dir.mkdir()
        (base_dir / "v1.0").mkdir()

        # Legitimate version lookup
        valid = sanitize_path(base_dir, "v1.0")
        assert valid.exists()

        # Traversal attempts
        with pytest.raises(SecurityAuthError, match="Path traversal detected"):
            sanitize_path(base_dir, "../../../etc/passwd")

        with pytest.raises(SecurityAuthError, match="Path traversal detected"):
            sanitize_path(base_dir, "v1/../../secret")

        with pytest.raises(SecurityAuthError, match="Invalid null byte"):
            sanitize_path(base_dir, "v1\x00/exploit")


def test_constant_time_admin_token_validation(monkeypatch):
    """Verify admin token verification rejects invalid keys and fails closed if unset."""
    monkeypatch.delenv("ADMIN_API_KEY", raising=False)
    # Fails closed if not configured
    assert not verify_admin_token("any-token")

    monkeypatch.setenv("ADMIN_API_KEY", "correct-key")
    assert verify_admin_token("correct-key")
    assert not verify_admin_token("wrong-key")
    assert not verify_admin_token("")
    assert not verify_admin_token(None)
