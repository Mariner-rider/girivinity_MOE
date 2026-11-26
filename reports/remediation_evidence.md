# Girivinity Test & Remediation Evidence Report

- **Generated (UTC):** `2026-11-26T10:41:41.597371+00:00`
- **Overall Required Suite Status:** `PASS`
- **Commit SHA:** `unavailable (detached/no-git-repo)`
- **Python:** `3.14.5`
- **PyTorch:** `not_installed`
- **Platform:** `Windows-11-10.0.26200-SP0`
- **Mode:** `sandbox`

## Executed Test Suites

| Suite | Category | File | Executed | Passed | Failed | Errors | Skipped | Exit Code | Status |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `k3_api_security` | `security` | `tests/test_api_security.py` | 20 | 20 | 0 | 0 | 0 | 0 | **PASS** |
| `k4_training_data` | `cpu_correctness` | `tests/test_k4_training_data.py` | 6 | 6 | 0 | 0 | 0 | 0 | **PASS** |
| `k5_checkpoints_and_gates` | `checkpoint_and_gates` | `tests/test_k5_checkpoints_and_gates.py` | 4 | 4 | 0 | 0 | 0 | 0 | **PASS** |
| `k6_evidence_and_ci` | `ci_evidence` | `tests/test_k6_evidence_and_ci.py` | 2 | 2 | 0 | 0 | 0 | 0 | **PASS** |

## Blocked / Not Run Gates

| Gate | Category | Reference | Status | Reason |
|---|---|---|---|---|
| `cuda_engine` | `optional_cuda_fsdp_performance` | `tests/test_cuda_engine.py` | **BLOCKED** | Requires CUDA GPU hardware (not available in CPU-only runner/sandbox). |
| `multi_rank_fsdp_checkpoint_resume` | `optional_cuda_fsdp_performance` | `tests/test_k5_checkpoints_and_gates.py` | **BLOCKED** | Requires multi-GPU torch.distributed process group. |
| `github_actions_remote_execution` | `ci_workflow` | `.github/workflows/ci.yml` | **NOT RUN** | Workflow file authored locally; remote execution occurs only when GitHub Actions runs it. |
