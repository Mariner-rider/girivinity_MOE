"""Fail-closed test runner and machine-readable evidence generator for Girivinity.

Executes configured test suites, verifies that zero-test collection, BOM/encoding
errors, parse errors, or unexpected skips never pass silently, and emits a
machine-readable JSON report plus a Markdown summary derived strictly from
observed command outputs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

SUITES: dict[str, list[dict[str, str]]] = {
    "sandbox": [
        {
            "name": "k3_api_security",
            "category": "security",
            "file": "tests/test_api_security.py",
        },
        {
            "name": "k4_training_data",
            "category": "cpu_correctness",
            "file": "tests/test_k4_training_data.py",
        },
        {
            "name": "k5_checkpoints_and_gates",
            "category": "checkpoint_and_gates",
            "file": "tests/test_k5_checkpoints_and_gates.py",
        },
        {
            "name": "k6_evidence_and_ci",
            "category": "ci_evidence",
            "file": "tests/test_k6_evidence_and_ci.py",
        },
    ],
    "ci": [
        {
            "name": "k3_api_security",
            "category": "security",
            "file": "tests/test_api_security.py",
        },
        {
            "name": "k4_training_data",
            "category": "cpu_correctness",
            "file": "tests/test_k4_training_data.py",
        },
        {
            "name": "k5_checkpoints_and_gates",
            "category": "checkpoint_and_gates",
            "file": "tests/test_k5_checkpoints_and_gates.py",
        },
        {
            "name": "k6_evidence_and_ci",
            "category": "ci_evidence",
            "file": "tests/test_k6_evidence_and_ci.py",
        },
        {
            "name": "k2_attention_mla",
            "category": "model_numerical_correctness",
            "file": "tests/test_mla.py",
        },
        {
            "name": "k2_attention_architecture",
            "category": "model_numerical_correctness",
            "file": "tests/test_architecture.py",
        },
        {
            "name": "k2_qk_norm_rope",
            "category": "model_numerical_correctness",
            "file": "tests/test_qk_norm_and_rope_scaling.py",
        },
    ],
}

BLOCKED_HARDWARE_SUITES = [
    {
        "name": "cuda_engine",
        "category": "optional_cuda_fsdp_performance",
        "file": "tests/test_cuda_engine.py",
        "status": "BLOCKED",
        "reason": "Requires CUDA GPU hardware (not available in CPU-only runner/sandbox).",
    },
    {
        "name": "multi_rank_fsdp_checkpoint_resume",
        "category": "optional_cuda_fsdp_performance",
        "file": "tests/test_k5_checkpoints_and_gates.py",
        "status": "BLOCKED",
        "reason": "Requires multi-GPU torch.distributed process group.",
    },
    {
        "name": "github_actions_remote_execution",
        "category": "ci_workflow",
        "file": ".github/workflows/ci.yml",
        "status": "NOT RUN",
        "reason": "Workflow file authored locally; remote execution occurs only when GitHub Actions runs it.",
    },
]


def _get_git_sha() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    except Exception:
        pass
    return "unavailable (detached/no-git-repo)"


def _get_torch_version() -> str:
    try:
        import torch  # type: ignore

        return getattr(torch, "__version__", "unknown")
    except Exception:
        return "not_installed"


def _parse_unittest_counts(raw_output: str) -> dict[str, int]:
    """Parse unittest summary lines fail-closed."""
    ran_match = re.search(r"Ran (\d+) tests? in", raw_output)
    executed = int(ran_match.group(1)) if ran_match else 0

    failed = 0
    errors = 0
    skipped = 0

    fail_match = re.search(r"failures=(\d+)", raw_output)
    if fail_match:
        failed = int(fail_match.group(1))

    err_match = re.search(r"errors=(\d+)", raw_output)
    if err_match:
        errors = int(err_match.group(1))

    skip_match = re.search(r"skipped=(\d+)", raw_output)
    if skip_match:
        skipped = int(skip_match.group(1))

    passed = max(0, executed - failed - errors - skipped)
    return {
        "executed": executed,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "skipped": skipped,
    }


def _run_suite(suite: dict[str, str], python_exe: str) -> dict[str, Any]:
    cmd = [python_exe, "-m", "unittest", suite["file"], "-v"]
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    raw_output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    counts = _parse_unittest_counts(raw_output)

    # Fail closed if zero tests collected, nonzero exit, failures, errors, or skips
    zero_collected = counts["executed"] == 0
    has_skips = counts["skipped"] > 0
    ok = (
        proc.returncode == 0
        and not zero_collected
        and counts["failed"] == 0
        and counts["errors"] == 0
        and not has_skips
    )

    status = "PASS" if ok else "FAIL"
    if zero_collected and proc.returncode == 0:
        status = "FAIL (zero tests collected)"
    elif has_skips:
        status = "BLOCKED/INCOMPLETE (skipped tests)"

    return {
        "name": suite["name"],
        "category": suite["category"],
        "file": suite["file"],
        "command": " ".join(cmd),
        "exit_code": proc.returncode,
        "status": status,
        "counts": counts,
        "raw_output": raw_output.strip(),
    }


def generate_evidence(mode: str, output_json: Path, output_md: Path) -> int:
    python_exe = sys.executable
    env_info = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "torch_version": _get_torch_version(),
        "platform": platform.platform(),
        "commit_sha": _get_git_sha(),
        "mode": mode,
    }

    results = [_run_suite(suite, python_exe) for suite in SUITES[mode]]
    overall_pass = all(r["status"] == "PASS" for r in results)

    report = {
        "schema_version": "1.0",
        "overall_status": "PASS" if overall_pass else "FAIL",
        "environment": env_info,
        "executed_suites": results,
        "blocked_or_not_run": BLOCKED_HARDWARE_SUITES,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    # Write UTF-8 without BOM
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Build Markdown summary strictly from observed results
    lines = [
        "# Girivinity Test & Remediation Evidence Report",
        "",
        f"- **Generated (UTC):** `{env_info['timestamp_utc']}`",
        f"- **Overall Required Suite Status:** `{report['overall_status']}`",
        f"- **Commit SHA:** `{env_info['commit_sha']}`",
        f"- **Python:** `{env_info['python_version']}`",
        f"- **PyTorch:** `{env_info['torch_version']}`",
        f"- **Platform:** `{env_info['platform']}`",
        f"- **Mode:** `{mode}`",
        "",
        "## Executed Test Suites",
        "",
        "| Suite | Category | File | Executed | Passed | Failed | Errors | Skipped | Exit Code | Status |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in results:
        c = r["counts"]
        lines.append(
            f"| `{r['name']}` | `{r['category']}` | `{r['file']}` | "
            f"{c['executed']} | {c['passed']} | {c['failed']} | {c['errors']} | {c['skipped']} | "
            f"{r['exit_code']} | **{r['status']}** |"
        )

    lines.extend(
        [
            "",
            "## Blocked / Not Run Gates",
            "",
            "| Gate | Category | Reference | Status | Reason |",
            "|---|---|---|---|---|",
        ]
    )
    for b in BLOCKED_HARDWARE_SUITES:
        lines.append(
            f"| `{b['name']}` | `{b['category']}` | `{b['file']}` | **{b['status']}** | {b['reason']} |"
        )

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return 0 if overall_pass else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Girivinity tests and generate fail-closed evidence.")
    parser.add_argument("--mode", choices=sorted(SUITES.keys()), default="sandbox")
    parser.add_argument("--output-json", default="reports/test_evidence.json")
    parser.add_argument("--output-md", default="reports/remediation_evidence.md")
    args = parser.parse_args()

    return generate_evidence(
        mode=args.mode,
        output_json=REPO_ROOT / args.output_json,
        output_md=REPO_ROOT / args.output_md,
    )


if __name__ == "__main__":
    raise SystemExit(main())
