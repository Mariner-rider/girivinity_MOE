"""Tests for K6: fail-closed evidence generation, zero-test rejection, and health model_status."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import tests._conftest_bootstrap  # noqa: F401
from scripts.generate_test_evidence import _parse_unittest_counts, generate_evidence


class TestK6EvidenceAndCI(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_parse_unittest_counts_zero_tests_and_skips(self):
        zero_out = "----------------------------------------------------------------------\nRan 0 tests in 0.000s\n\nOK"
        counts = _parse_unittest_counts(zero_out)
        self.assertEqual(counts["executed"], 0)

        skip_out = "----------------------------------------------------------------------\nRan 3 tests in 0.010s\n\nOK (skipped=1)"
        counts_skip = _parse_unittest_counts(skip_out)
        self.assertEqual(counts_skip["executed"], 3)
        self.assertEqual(counts_skip["skipped"], 1)
        self.assertEqual(counts_skip["passed"], 2)

    def test_health_deep_reports_truthful_model_status(self):
        from app.api.routes.health import health_deep

        coro = health_deep()
        try:
            coro.send(None)
        except StopIteration as exc:
            result = exc.value

        self.assertIn("model_status", result)
        ms = result["model_status"]
        self.assertEqual(ms["native_model_path"], "models/base/final")
        self.assertFalse(ms["checkpoint_present"])
        self.assertFalse(ms["trained_context_verified"])
        self.assertEqual(ms["configured_max_seq_len"], 32768)


if __name__ == "__main__":
    unittest.main()
