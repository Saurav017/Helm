# Contributing to Helm

Thank you for considering contributing to Helm! This document provides guidelines and expectations for contributors.

---

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). By participating, you agree to uphold this code.

---

## Getting Started

### Prerequisites
- Python ≥ 3.11
- `git`
- Familiarity with [INSTRUCTIONS.md](INSTRUCTIONS.md) — **read it before writing any code**

### Development Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_ORG/helm.git
cd helm

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Verify setup
ruff check .
mypy --strict helm/
pytest
```

---

## Development Workflow

### 1. Find or Create an Issue
- Check existing issues before starting work.
- For new features, open an issue for discussion before writing code.
- For bugs, include reproduction steps.

### 2. Branch
```bash
git checkout -b feat/your-feature-name
# or: fix/bug-description, refactor/module-name
```

### 3. Develop
- Follow the coding standards in [INSTRUCTIONS.md §2](INSTRUCTIONS.md#2-coding-standards).
- Write tests alongside your code — PRs without tests for new functionality will not be merged.
- Run the quality checks frequently:
  ```bash
  ruff format .
  ruff check . --fix
  mypy --strict helm/
  pytest
  ```

### 4. Commit
Follow [Conventional Commits](https://www.conventionalcommits.org/):
```
feat(core): add task dependency resolution to scheduler
fix(mcp): handle malformed JSON-RPC responses gracefully
test(workers): add integration test for coder worker
docs: add tool pruning architecture deep-dive
```

### 5. Submit a Pull Request
- Target the `main` branch.
- Fill out the PR template completely.
- Ensure CI passes (lint, type check, tests).
- Request review from a maintainer.

---

## Architecture Decisions

If your change affects the architecture (new modules, new dependencies, new patterns), please:
1. Document the rationale in the PR description.
2. Update relevant docs in `docs/`.
3. Ensure compliance with architecture invariants in [INSTRUCTIONS.md §5](INSTRUCTIONS.md#5-architecture-invariants).

Adding a new dependency requires explicit justification against the approved list in INSTRUCTIONS.md §1.3.

---

## Testing Standards

- **Unit tests** for all pure logic (state transitions, circuit breaker rules, context masking).
- **Integration tests** for MCP client ↔ server communication and worker pipelines.
- **E2E tests** for CLI commands.
- Minimum coverage target: **80%** for new code.
- Test files mirror source structure: `helm/core/supervisor.py` → `tests/unit/test_supervisor.py`.

---

## Questions?

Open a Discussion on GitHub or reach out to the maintainers. We're happy to help you contribute.
