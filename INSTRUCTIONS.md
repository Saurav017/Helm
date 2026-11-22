# INSTRUCTIONS.md — Helm Agent Governance & Session Protocol

> **Authority level: IMMUTABLE.**
> This document is the canonical instruction set for any AI coding agent working on the Helm codebase.
> It MUST be read at the start of every session. It MUST NOT be modified without explicit human approval.
> Violations of these constraints constitute a session failure.

---

## 1. Project Identity & Philosophy

### 1.1 What Is Helm?
Helm (`helmx`) is an open-source, terminal-first agentic orchestration runtime and multi-agent control plane. It decomposes complex goals into a supervised DAG of worker tasks, communicates with tools via the Model Context Protocol (MCP), and enforces deterministic safety guardrails at every layer.

### 1.2 Core Philosophy

| Principle | Mandate |
|---|---|
| **Terminal-first** | Helm is a CLI tool. No web UI is required. All interaction happens through the terminal using Rich/Textual rendering. Every feature must be usable via `helmx` commands. |
| **Zero unnecessary dependencies** | Every `pip install` addition must be justified. Prefer stdlib. Avoid meta-frameworks (LangChain, CrewAI, etc.) — Helm IS the framework. |
| **Deterministic safety over unbounded autonomy** | An agent that does nothing is safer than an agent that does the wrong thing. Default to requiring confirmation. Default to failing closed. |
| **Observability is non-negotiable** | Every state transition, tool call, model invocation, and error MUST be loggable and traceable. Silent failures are forbidden. |
| **Composition over inheritance** | Prefer protocols (`typing.Protocol`), dependency injection, and composition. Avoid deep class hierarchies. |

### 1.3 Approved Core Dependencies
The following are the **only** pre-approved external dependencies. Any addition requires justification in the PR description.

| Dependency | Purpose | Constraint |
|---|---|---|
| `typer` | CLI framework | Required |
| `rich` | Terminal rendering | Required |
| `pydantic` (v2+) | Data validation, settings, state schemas | Required |
| `httpx` | Async HTTP client (MCP SSE transport) | Required |
| `openai` / `anthropic` / `google-genai` | LLM provider SDKs | At least one required; others optional |
| `numpy` | Embedding math (cosine similarity) | Required for tool pruning |
| `pytest` | Testing | Dev dependency |
| `ruff` | Linting & formatting | Dev dependency |
| `mypy` | Static type checking | Dev dependency |

---

## 2. Coding Standards

### 2.1 Language & Runtime
- **Python ≥ 3.11** — use modern syntax (`match`, `type` aliases, `ExceptionGroup`).
- **Async-first** — all I/O-bound operations (LLM calls, MCP transport, file I/O) MUST be `async`. Use `asyncio` exclusively; no `threading` for I/O.
- **Type hints everywhere** — every function signature, every variable where the type is non-obvious. Use `typing.Protocol` for interfaces.

### 2.2 Pydantic v2 Models
- All data schemas (state, config, task briefs, tool results) MUST be Pydantic `BaseModel` subclasses.
- Use `model_validator`, `field_validator` for invariants.
- Serialize/deserialize with `.model_dump()` / `.model_validate()` — never raw `dict` access on structured data.

### 2.3 Error Handling
```
RULE: No silent failures. Ever.
```
- Every `except` block MUST either:
  1. Log the error with full context and re-raise, OR
  2. Return a typed error result (e.g., `TaskResult(status=TaskStatus.FAILED, error=...)`)
- Bare `except:` and `except Exception:` with `pass` are **forbidden**.
- Use custom exception classes inheriting from a `HelmError` base.

### 2.4 Logging
- Use `structlog` or stdlib `logging` with structured key-value output.
- Every log line MUST include: `event`, `component`, and relevant IDs (`task_id`, `worker_id`, `tool_name`).
- Log levels: `DEBUG` for internals, `INFO` for state transitions, `WARNING` for recoverable issues, `ERROR` for failures, `CRITICAL` for safety violations.

### 2.5 Formatting & Linting
- **Formatter:** `ruff format` (Black-compatible, 100-char line width).
- **Linter:** `ruff check` with `select = ["ALL"]` and targeted `ignore` list.
- **Type checker:** `mypy --strict` must pass with zero errors.
- All three checks are enforced in CI. Code that fails any check MUST NOT be merged.

---

## 3. Agent Execution Constraints

These are inviolable rules governing how the Helm runtime and its sub-agents operate.

### 3.1 Model Context Protocol (MCP) Compliance
- **Every external tool** MUST be accessed through an MCP-compliant interface (JSON-RPC 2.0 over `stdio` or `HTTP-SSE`).
- Tool schemas MUST conform to the MCP `Tool` type: `name`, `description`, `inputSchema` (JSON Schema).
- The system MUST NOT call tools via ad-hoc subprocess invocations, raw HTTP calls, or SDK-specific methods outside the MCP transport layer.

### 3.2 Terminal Safety Gate (HITL)
- **Every mutating action** — file writes, file deletions, bash/shell execution, git commits, git pushes, API calls with side effects — MUST be gated through the interactive terminal confirmation mechanism.
- The gate MUST present: the tool name, a human-readable summary of what will change, and a `[y/N]` prompt.
- Default answer is **N** (deny). Timeout defaults to deny.
- The gate may be configured to auto-approve specific tool categories (e.g., `auto_approve_reads = true`), but this MUST be an explicit user configuration, never a default.
- **Bypass is forbidden in code.** There is no programmatic way to skip the safety gate. The only path is user input.

### 3.3 Context Isolation for Workers
```
RULE: Worker agents MUST NEVER receive raw full-chat context.
```
- When the Supervisor dispatches a task to a Worker, it MUST construct an **isolated task brief** containing:
  1. The specific sub-task description.
  2. Relevant extracted context (code snippets, file paths, prior results) — only what is needed.
  3. The list of available tools (post-pruning).
  4. Constraints (max steps, budget slice).
- The full conversation history, other workers' results, and the Supervisor's internal reasoning are **excluded** from the brief.
- The Context Masking Engine is responsible for this distillation.

### 3.4 State Schema & Circuit Breaker
- **All state transitions** (task created, task assigned, tool called, result received, task completed/failed) MUST update the strongly-typed `SessionState` Pydantic model.
- Every tool invocation MUST pass through the Circuit Breaker, which checks:
  1. **Duplicate detection:** SHA-256 hash of `(tool_name, canonical_arguments)`. If a hash collision is found, the call is blocked and logged.
  2. **Step ceiling:** If the current task has exceeded `max_steps`, the call is blocked and the task is force-terminated with a summary.
  3. **Budget cap:** If cumulative token cost exceeds the configured budget, the call is blocked.
- The Circuit Breaker MUST log every blocked call with the reason.

### 3.5 Tool Pruning Protocol
- The system MUST NOT inject all available tool schemas into every LLM prompt.
- Before each LLM call, the Tool Pruner:
  1. Embeds the current task/query.
  2. Performs cosine similarity search against the tool description index.
  3. Returns the **top-k** (default k=5) most relevant tool schemas.
- Only these pruned schemas are included in the LLM request.
- The full tool registry is never sent to any LLM.

### 3.6 Tiered Model Routing
- The Model Router selects the appropriate model tier based on task classification:
  - **Fast tier:** Triage, classification, simple summarization, yes/no decisions.
  - **Standard tier:** Code generation, structured output, moderate reasoning.
  - **Deep tier:** Multi-step planning, architecture design, complex debugging.
- Tier assignment is determined by the Supervisor during task decomposition.
- Workers MUST use the assigned tier unless the Supervisor explicitly overrides.

---

## 4. Session Continuity Protocol

Every agent session MUST follow this startup sequence before making any modifications:

### 4.1 Session Initialization Checklist
```
1. READ this file (INSTRUCTIONS.md) in its entirety.
2. READ README.md for project context and current roadmap status.
3. INSPECT the current state of the codebase:
   a. List the top-level directory structure.
   b. Check for any existing implementation in files you plan to modify.
   c. Review recent git log (last 10 commits) to understand recent changes.
4. RUN existing tests (`pytest`) to establish a green baseline.
   - If tests fail, STOP and report failures before making changes.
5. REVIEW any open TODOs or FIXMEs in the files you plan to modify.
6. ONLY THEN proceed with the assigned task.
```

### 4.2 Pre-Commit Checklist
Before considering any task complete, the agent MUST:
```
1. RUN `ruff format --check .` — fix any formatting issues.
2. RUN `ruff check .` — fix any lint violations.
3. RUN `mypy --strict helm/` — fix any type errors.
4. RUN `pytest` — all tests must pass, including new tests for new code.
5. UPDATE docstrings for any modified public APIs.
6. VERIFY that INSTRUCTIONS.md constraints have not been violated.
```

### 4.3 Incremental Development Rules
- **Never rewrite files from scratch** unless explicitly instructed. Read existing content first, then make targeted modifications.
- **Never remove existing tests** unless they test deleted functionality.
- **Never add dependencies** without documenting the justification.
- **Always preserve existing docstrings and comments** unless they are factually wrong or reference deleted code.
- **If unsure, ask.** Do not guess at architectural decisions. Halt and request clarification.

### 4.4 Git Conventions
- Branch naming: `feat/short-description`, `fix/short-description`, `refactor/short-description`
- Commit messages: [Conventional Commits](https://www.conventionalcommits.org/) format.
  ```
  feat(core): add circuit breaker duplicate hash detection
  fix(mcp): handle connection timeout in stdio transport
  refactor(cli): extract safety gate into standalone module
  test(workers): add unit tests for coder worker task brief
  docs: update README roadmap checkboxes
  ```
- Atomic commits: one logical change per commit.

---

## 5. Architecture Invariants

These are structural properties of the system that MUST be maintained across all changes:

1. **The Supervisor is the only component that talks to workers.** Workers do not communicate with each other directly.
2. **The CLI layer (`helm/cli/`) never imports from `helm/core/` internals.** It interacts through a defined public API surface.
3. **MCP transports are pluggable.** Adding a new transport MUST NOT require modifying existing transport code.
4. **State is the single source of truth.** No component maintains shadow state. All reads/writes go through `SessionState`.
5. **The Circuit Breaker sits between the Supervisor and the MCP layer.** It cannot be bypassed by workers or the CLI.
6. **Workers are stateless between tasks.** Any persistent state lives in `SessionState`, not in worker instances.
7. **Configuration flows one way:** config files → `Settings` model → components. Components never write back to config files.

---

## 6. Security & Trust Boundaries

```
┌─────────────────────────────────────────────────┐
│  TRUSTED ZONE (user's terminal)                 │
│  • CLI input/output                             │
│  • Safety Gate confirmation                     │
│  • Configuration files                          │
├─────────────────────────────────────────────────┤
│  SUPERVISED ZONE (Helm runtime)                 │
│  • Supervisor orchestration logic               │
│  • State management                             │
│  • Circuit Breaker enforcement                  │
│  • Context Masking                              │
├─────────────────────────────────────────────────┤
│  UNTRUSTED ZONE (LLM outputs & tool results)   │
│  • All LLM-generated content                    │
│  • All tool execution results                   │
│  • Worker sub-agent outputs                     │
└─────────────────────────────────────────────────┘
```

- LLM outputs are **never trusted**. They are validated, logged, and gated before execution.
- Tool results are **validated against expected schemas** before being incorporated into state.
- Worker outputs are **reviewed by the Supervisor** before being surfaced to the user.

---

## 7. File Ownership Map

| Path | Owner | Purpose |
|---|---|---|
| `helm/cli/` | CLI Team | Terminal interface, rendering, safety gate |
| `helm/core/` | Core Team | Supervisor, scheduler, state, circuit breaker |
| `helm/protocols/` | Protocol Team | MCP client/host, tool registry |
| `helm/memory/` | Memory Team | Embeddings, tool index, pruner |
| `helm/workers/` | Worker Team | Base worker, specialized agents |
| `helm/config/` | Config Team | Settings schema, defaults |
| `tests/` | All | Test coverage for respective modules |
| `INSTRUCTIONS.md` | **Project Lead only** | Governance — requires explicit approval to modify |

---

*Last updated: 2026-11-07*
*Document version: 1.0.0*
