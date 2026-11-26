# Branch Protection & Merge Policy (Administrator Guide)

> **Important:** GitHub branch protection rules cannot be enabled by committing source files. A repository administrator must apply the settings below in the GitHub repository settings UI.

## Required Settings for `main`

Navigate to **GitHub Repository -> Settings -> Branches -> Add branch protection rule** for branch name pattern `main`:

1. **Require a pull request before merging**
   - Enable **Require approvals** (at least 1 approving review from a code owner or maintainer).
   - Enable **Dismiss stale pull request approvals when new commits are pushed**.
2. **Require status checks to pass before merging**
   - Enable **Require branches to be up to date before merging**.
   - Add the following required checks from `.github/workflows/ci.yml`:
     - `Lint and Compile Checks` (`lint-and-compile`)
     - `CPU Correctness, Security & Checkpoint Gates` (`cpu-unit-and-security`)
3. **Restrict history rewrites and deletion**
   - Disable **Allow force pushes** (must remain unchecked).
   - Disable **Allow deletions** (must remain unchecked).
4. **Do not bypass**
   - Enable **Do not allow bypassing the above settings** for administrators where policy permits.

## Optional / Hardware-Gated Checks

- `Integration Tests (PostgreSQL + API)` (`integration-db-api`) requires a PostgreSQL service container and is run as a separate CI job.
- `Optional CUDA / Multi-Rank FSDP` (`gpu-fsdp-performance`) is triggered manually (`workflow_dispatch`) on self-hosted GPU runners and must never be reported as passing on CPU-only runners.
