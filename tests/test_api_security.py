"""Tests for API authentication, route authorization inventory, and privileged access control."""

from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch

import tests._conftest_bootstrap  # noqa: F401
from fastapi import HTTPException
from starlette.requests import Request

from app.main import app
from app.security.api_auth import (
    ROUTE_CLASSIFICATION,
    AccessTier,
    Principal,
    Role,
    _authenticate_credential,
    _constant_time_in,
    _extract_credential,
    _is_revoked,
    enforce_public_chat_rate_limit,
    get_correlation_id,
    log_audit_event,
    require_admin,
    require_authenticated_user,
    require_roles,
    require_service_or_admin,
    reset_public_chat_rate_limits,
)


def run_sync(coro):
    """Run an async coroutine synchronously without spawning background threads or event loops."""
    try:
        coro.send(None)
    except StopIteration as exc:
        return exc.value


class TestRouteInventory(unittest.TestCase):
    """Enumerate every live runtime registered route and verify it is explicitly classified."""

    def test_all_live_routes_are_classified(self):
        live_routes: set[tuple[str, str]] = set()
        for route in app.routes:
            methods = getattr(route, "methods", None)
            path = getattr(route, "path", None)
            if methods and path:
                for method in methods:
                    if method not in {"OPTIONS", "HEAD"}:
                        live_routes.add((method, path))

        classified_routes = set(ROUTE_CLASSIFICATION.keys())
        unclassified = live_routes - classified_routes
        self.assertEqual(
            unclassified,
            set(),
            f"Found live routes missing from ROUTE_CLASSIFICATION: {unclassified}",
        )

    def test_no_phantom_routes_in_classification(self):
        live_routes: set[tuple[str, str]] = set()
        for route in app.routes:
            methods = getattr(route, "methods", None)
            path = getattr(route, "path", None)
            if methods and path:
                for method in methods:
                    if method not in {"OPTIONS", "HEAD"}:
                        live_routes.add((method, path))

        classified_routes = set(ROUTE_CLASSIFICATION.keys())
        phantom_routes = classified_routes - live_routes
        self.assertEqual(
            phantom_routes,
            set(),
            f"ROUTE_CLASSIFICATION contains routes not present in live app: {phantom_routes}",
        )

    def test_privileged_routes_have_authorization_dependencies(self):
        """Verify that every route classified as privileged attaches security dependencies."""
        endpoints_to_deps: dict[tuple[str, str], list[str]] = {}
        for r in app.routes:
            methods = getattr(r, "methods", None)
            path = getattr(r, "path", None)
            dependant = getattr(r, "dependant", None)
            deps = []
            if dependant:
                deps = [d.call.__name__ for d in dependant.dependencies if hasattr(d, "call")]
            if methods and path:
                for m in methods:
                    if m not in {"OPTIONS", "HEAD"}:
                        endpoints_to_deps[(m, path)] = deps

        for (method, path), tier in ROUTE_CLASSIFICATION.items():
            deps = endpoints_to_deps.get((method, path), [])
            if tier in {AccessTier.ADMIN_ONLY, AccessTier.SERVICE_INTERNAL, AccessTier.AUTHENTICATED_USER}:
                self.assertTrue(
                    len(deps) > 0,
                    f"Privileged route {method} {path} (Tier: {tier}) has NO dependencies registered!",
                )
            elif tier == AccessTier.PUBLIC_CHAT:
                self.assertIn(
                    "enforce_public_chat_rate_limit",
                    deps,
                    f"Public chat route {method} {path} missing rate limit dependency",
                )


class TestApiAuthMechanisms(unittest.TestCase):
    """Test core credential extraction, constant time check, revocation, and role mapping."""

    def setUp(self):
        self.env_patch = patch.dict(
            os.environ,
            {
                "GIRIVINITY_ADMIN_API_KEYS": "admin-secret-token-123",
                "GIRIVINITY_SERVICE_API_KEYS": "service-secret-token-456",
                "GIRIVINITY_USER_API_KEYS": "user-secret-token-789",
                "RASP_API_KEYS": "rasp-internal-token-999",
                "GIRIVINITY_REVOKED_KEYS": "revoked-token-000",
            },
            clear=False,
        )
        self.env_patch.start()

    def tearDown(self):
        self.env_patch.stop()

    def test_extract_credential_bearer(self):
        token, method = _extract_credential("Bearer token123", None, None)
        self.assertEqual(token, "token123")
        self.assertEqual(method, "bearer")

    def test_extract_credential_x_api_key(self):
        token, method = _extract_credential(None, "key123", None)
        self.assertEqual(token, "key123")
        self.assertEqual(method, "x-api-key")

    def test_extract_credential_x_rasp_key(self):
        token, method = _extract_credential(None, None, "rasp123")
        self.assertEqual(token, "rasp123")
        self.assertEqual(method, "x-rasp-key")

    def test_extract_credential_none(self):
        self.assertIsNone(_extract_credential(None, None, None))

    def test_extract_credential_malformed_bearer(self):
        with self.assertRaises(HTTPException) as cm:
            _extract_credential("Basic 12345", None, None)
        self.assertEqual(cm.exception.status_code, 401)

    def test_authenticate_admin_key(self):
        p = _authenticate_credential("admin-secret-token-123", "x-api-key")
        self.assertIsNotNone(p)
        self.assertEqual(p.role, Role.ADMIN)

    def test_authenticate_service_key(self):
        p = _authenticate_credential("service-secret-token-456", "x-api-key")
        self.assertIsNotNone(p)
        self.assertEqual(p.role, Role.SERVICE)

    def test_authenticate_user_key(self):
        p = _authenticate_credential("user-secret-token-789", "bearer")
        self.assertIsNotNone(p)
        self.assertEqual(p.role, Role.USER)

    def test_forbidden_default_placeholder_keys_rejected(self):
        for bad_key in ["default", "secret", "admin", "password", "test", ""]:
            self.assertIsNone(_authenticate_credential(bad_key, "bearer"))

    def test_revoked_key_rejected(self):
        self.assertIsNone(_authenticate_credential("revoked-token-000", "x-api-key"))


class TestRolePolicyDependencies(unittest.TestCase):
    """Test require_roles dependencies directly and synchronously."""

    def setUp(self):
        self.env_patch = patch.dict(
            os.environ,
            {
                "GIRIVINITY_ADMIN_API_KEYS": "admin-key-abc",
                "GIRIVINITY_SERVICE_API_KEYS": "service-key-xyz",
                "GIRIVINITY_USER_API_KEYS": "user-key-123",
                "RASP_API_KEYS": "rasp-key-456",
            },
            clear=False,
        )
        self.env_patch.start()
        reset_public_chat_rate_limits()

    def tearDown(self):
        self.env_patch.stop()

    def _make_request(self, path: str = "/test"):
        scope = {
            "type": "http",
            "method": "POST",
            "path": path,
            "headers": [],
            "client": ("127.0.0.1", 80),
        }
        return Request(scope)

    def test_require_admin_missing_credentials(self):
        dep = require_admin
        req = self._make_request("/admin/notifications")
        with self.assertRaises(HTTPException) as cm:
            run_sync(dep(req, authorization=None, x_api_key=None, x_rasp_key=None))
        self.assertEqual(cm.exception.status_code, 401)

    def test_require_admin_forbidden_for_user_role(self):
        dep = require_admin
        req = self._make_request("/admin/notifications")
        with self.assertRaises(HTTPException) as cm:
            run_sync(dep(req, authorization="Bearer user-key-123", x_api_key=None, x_rasp_key=None))
        self.assertEqual(cm.exception.status_code, 403)

    def test_require_admin_forbidden_for_service_role(self):
        dep = require_admin
        req = self._make_request("/admin/notifications")
        with self.assertRaises(HTTPException) as cm:
            run_sync(dep(req, authorization=None, x_api_key="service-key-xyz", x_rasp_key=None))
        self.assertEqual(cm.exception.status_code, 403)

    def test_require_admin_success(self):
        dep = require_admin
        req = self._make_request("/admin/notifications")
        p = run_sync(dep(req, authorization=None, x_api_key="admin-key-abc", x_rasp_key=None))
        self.assertIsInstance(p, Principal)
        self.assertEqual(p.role, Role.ADMIN)

    def test_require_service_or_admin_allows_service_and_admin(self):
        dep = require_service_or_admin
        req = self._make_request("/modules")
        # User fails with 403
        with self.assertRaises(HTTPException) as cm:
            run_sync(dep(req, authorization="Bearer user-key-123", x_api_key=None, x_rasp_key=None))
        self.assertEqual(cm.exception.status_code, 403)

        # Service succeeds
        p_srv = run_sync(dep(req, authorization=None, x_api_key="service-key-xyz", x_rasp_key=None))
        self.assertEqual(p_srv.role, Role.SERVICE)

        # Admin succeeds
        p_adm = run_sync(dep(req, authorization=None, x_api_key="admin-key-abc", x_rasp_key=None))
        self.assertEqual(p_adm.role, Role.ADMIN)

    def test_require_authenticated_user_allows_user(self):
        dep = require_authenticated_user
        req = self._make_request("/v1/analyze/code")
        p = run_sync(dep(req, authorization="Bearer user-key-123", x_api_key=None, x_rasp_key=None))
        self.assertEqual(p.role, Role.USER)

    def test_public_chat_rate_limiting(self):
        with patch.dict(os.environ, {"GIRIVINITY_PUBLIC_CHAT_RPM": "2"}):
            reset_public_chat_rate_limits()
            req = self._make_request("/chat/message")
            enforce_public_chat_rate_limit(req)  # 1st request
            enforce_public_chat_rate_limit(req)  # 2nd request
            with self.assertRaises(HTTPException) as cm:
                enforce_public_chat_rate_limit(req)  # 3rd request should exceed RPM=2
            self.assertEqual(cm.exception.status_code, 429)


if __name__ == "__main__":
    unittest.main()
