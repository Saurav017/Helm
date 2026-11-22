<p align="center">
  <img src="docs/assets/helm-logo.png" alt="Helm Logo" width="200" />
</p>

<h1 align="center">Helm</h1>

<p align="center">
  <strong>Terminal-first agentic orchestration runtime & multi-agent control plane.</strong>
</p>

<p align="center">
  <!-- Badges — replace URLs once CI/PyPI are live -->
  <a href="https://github.com/YOUR_ORG/helm/actions"><img src="https://img.shields.io/github/actions/workflow/status/YOUR_ORG/helm/ci.yml?branch=main&label=CI&style=flat-square" alt="CI Status"></a>
  <a href="https://pypi.org/project/helmx/"><img src="https://img.shields.io/pypi/v/helmx?style=flat-square&color=blue" alt="PyPI Version"></a>
  <a href="https://pypi.org/project/helmx/"><img src="https://img.shields.io/pypi/pyversions/helmx?style=flat-square" alt="Python Versions"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-green?style=flat-square" alt="License"></a>
  <a href="https://github.com/YOUR_ORG/helm/stargazers"><img src="https://img.shields.io/github/stars/YOUR_ORG/helm?style=flat-square" alt="Stars"></a>
</p>

---

## Why Helm?

Modern AI agents suffer from three systemic failures:

| Problem | What happens | Helm's answer |
|---|---|---|
| **Context pollution** | Every tool schema, every past message goes into one giant prompt. Tokens explode, accuracy degrades. | **Context Isolation & Masking** — Workers receive only a distilled task brief, never the full conversation thread. |
| **Brittle autonomy** | Agents run wild: infinite loops, duplicate calls, destructive file writes with zero human oversight. | **Deterministic Guardrails** — Circuit breakers, duplicate-call hash detection, max-step ceilings, and a mandatory terminal HITL gate for all mutations. |
| **Flat execution** | Single-model, single-thread. No way to route cheap tasks to fast models and hard tasks to deep reasoners. | **Tiered Model Routing & DAG Scheduling** — Supervisor decomposes goals into a dependency-aware task graph routed across specialized workers and model tiers. |

Helm is not another wrapper around `openai.chat.completions`. It is a **production control plane** for multi-agent systems that speaks the **Model Context Protocol (MCP)** natively, prunes tool schemas via **RAG-based retrieval**, and never executes a destructive action without your explicit terminal confirmation.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         TERMINAL (Rich TUI)                      │
│  helmx "deploy the auth service to staging"                      │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │  ⚙ Safety Gate: [y/N] confirm before mutations            │   │
│  └────────────────────────────────────────────────────────────┘   │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                     SUPERVISOR (Core Orchestrator)                │
│                                                                  │
│  ┌──────────┐   ┌──────────────┐   ┌────────────────────────┐   │
│  │  Model    │   │  Task Graph  │   │  Circuit Breaker       │   │
│  │  Router   │──▶│  (DAG)       │──▶│  • Dup hash detect     │   │
│  │  (Tiered) │   │  Scheduler   │   │  • Max step ceiling    │   │
│  └──────────┘   └──────┬───────┘   │  • Cost accumulator    │   │
│                        │           └────────────────────────┘   │
│                        ▼                                        │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Context Masking Engine                                  │    │
│  │  Full thread ──▶ Isolated Task Brief per worker          │    │
│  └─────────────────────────────────────────────────────────┘    │
└──────────────────────┬───────────────────────────────────────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
   ┌────────────┐┌────────────┐┌────────────┐
   │  Worker:   ││  Worker:   ││  Worker:   │
   │  Coder     ││  Researcher││  DevOps    │
   │  (GPT-4o)  ││  (Flash)   ││  (Sonnet)  │
   └──────┬─────┘└──────┬─────┘└──────┬─────┘
          │             │             │
          ▼             ▼             ▼
   ┌─────────────────────────────────────────┐
   │        MCP Tool Layer (JSON-RPC)        │
   │                                          │
   │  ┌──────────┐ ┌───────┐ ┌────────────┐ │
   │  │ Filesystem│ │ Git   │ │ Browser    │ │
   │  │ Server    │ │ Server│ │ Server     │ │
   │  └──────────┘ └───────┘ └────────────┘ │
   │                                          │
   │  Tool Index (RAG Pruning)               │
   │  100+ schemas ──▶ top 3–5 per turn      │
   └─────────────────────────────────────────┘
```

---

## Core Capabilities

### 1. Supervisor–Worker DAG Scheduling
The Supervisor receives a user goal, decomposes it into a **directed acyclic graph** of sub-tasks, and schedules workers with dependency awareness. Workers report structured results back; the Supervisor synthesizes.

### 2. Native MCP Client / Host
Helm speaks [Model Context Protocol](https://modelcontextprotocol.io) over `stdio` and `HTTP-SSE` transports. Connect any MCP-compliant tool server — filesystem, database, browser, custom — without writing glue code.

### 3. Dynamic Tool Pruning (RAG for Tools)
Instead of injecting dozens of tool schemas into every prompt, Helm embeds tool descriptions into a lightweight vector index and retrieves only the **top 3–5 relevant schemas** per turn. Fewer tokens, better accuracy.

### 4. Context Isolation & Masking
Worker sub-agents never see the full conversation. The Context Masking Engine distills an **isolated task brief** containing only the information a worker needs, preventing context pollution and data leakage.

### 5. Deterministic Production Guardrails
| Guardrail | Mechanism |
|---|---|
| Duplicate call detection | SHA-256 hash of `(tool_name, arguments)` — blocks identical re-invocations |
| Max step ceiling | Configurable hard limit per task (default: 25 steps) |
| Cost accumulator | Running token/cost tally with configurable budget cap |
| **Terminal Safety Gate** | Every mutation (`file_write`, `bash_exec`, `git_commit`) requires explicit `[y/N]` confirmation in the terminal |

### 6. Tiered Model Routing
| Tier | Use Case | Example Models |
|---|---|---|
| **Fast** | Classification, triage, summarization | Gemini Flash, GPT-4o-mini, Haiku |
| **Standard** | Code generation, structured output | GPT-4o, Sonnet, Gemini Pro |
| **Deep** | Planning, architecture, complex reasoning | o1, o3, Opus |

---

## Quickstart

### Installation

```bash
# From PyPI (once published)
pip install helmx

# From source
git clone https://github.com/YOUR_ORG/helm.git
cd helm
pip install -e ".[dev]"
```

### Usage

```bash
# One-shot task execution
helmx "refactor the auth module to use dependency injection"

# Interactive multi-turn session
helmx --interactive

# List connected MCP tool servers and available tools
helmx tools

# Run with a specific model tier override
helmx --tier deep "design a migration plan for the payments database"

# Dry-run mode: plan the DAG without executing any tools
helmx --dry-run "deploy v2.1 to staging"

# Show current configuration
helmx config show

# Connect a new MCP server
helmx config add-server filesystem --command "npx @mcp/filesystem /path/to/project"
```

### Configuration

Helm reads configuration from (in priority order):
1. CLI flags
2. Environment variables (`HELMX_*`)
3. Project-local `helmx.toml`
4. User-global `~/.config/helmx/config.toml`

```toml
# helmx.toml — project-level configuration
[supervisor]
max_steps = 25
budget_usd = 2.00

[models]
fast = "gemini-2.0-flash"
standard = "gpt-4o"
deep = "o3"

[safety]
hitl_mutations = true       # Require [y/N] for all mutations
auto_approve_reads = true   # Skip confirmation for read-only tools

[[mcp_servers]]
name = "filesystem"
transport = "stdio"
command = "npx @mcp/filesystem ."

[[mcp_servers]]
name = "github"
transport = "sse"
url = "http://localhost:8080/sse"
```

---

## Comparison Matrix

| Capability | Raw LLM Script | LangGraph | CrewAI | **Helm** |
|---|:---:|:---:|:---:|:---:|
| Multi-agent DAG scheduling | ✗ | ◐ (manual graph) | ◐ (sequential) | **✓ (dynamic)** |
| Native MCP support | ✗ | ✗ | ✗ | **✓** |
| Tool pruning (RAG) | ✗ | ✗ | ✗ | **✓** |
| Context isolation per worker | ✗ | ✗ | ◐ | **✓** |
| Circuit breakers & guardrails | ✗ | ✗ | ✗ | **✓** |
| Terminal HITL safety gate | ✗ | ✗ | ✗ | **✓** |
| Tiered model routing | ✗ | ◐ | ◐ | **✓** |
| Zero framework lock-in | ✓ | ✗ | ✗ | **✓** |

---

## Project Structure

```
helm/
├── cli/                  # Typer CLI app, Rich rendering, Safety Gate
│   ├── commands/         # Subcommand handlers (run, tools, config, interactive)
│   └── rendering/        # Rich panels, spinners, themes
├── core/                 # Supervisor, DAG Scheduler, State, Circuit Breaker
├── protocols/            # MCP client/host, Tool Registry
│   └── mcp/              # MCP transports (stdio, SSE), type definitions
├── memory/               # Embedding engine, Tool index, RAG pruner
├── workers/              # Base worker + specialized agents (Coder, Researcher, DevOps)
└── config/               # Settings schema (Pydantic), defaults
tests/
├── unit/                 # Isolated component tests
├── integration/          # MCP client ↔ server, worker pipelines
└── e2e/                  # Full CLI end-to-end tests
configs/                  # Example configuration files
docs/                     # Architecture deep-dives and guides
```

---

## Roadmap

### Phase 1 — Foundation (Current)
- [x] Project structure, governance docs, instruction set
- [ ] Core state schema (Pydantic v2 models)
- [ ] Supervisor loop with single-worker execution
- [ ] MCP stdio transport client
- [ ] Terminal safety gate (HITL `[y/N]`)
- [ ] Basic CLI (`helmx "prompt"`)

### Phase 2 — Multi-Agent & Tool Intelligence
- [ ] DAG task scheduler with dependency resolution
- [ ] Worker sub-agents: Coder, Researcher, DevOps
- [ ] Context masking engine
- [ ] Tool embedding index + RAG pruner
- [ ] Tiered model router

### Phase 3 — Production Hardening
- [ ] Circuit breaker suite (dup detection, cost caps, step limits)
- [ ] MCP SSE transport
- [ ] Session persistence & replay
- [ ] Streaming token output in terminal
- [ ] Comprehensive test coverage (>80%)

### Phase 4 — Ecosystem
- [ ] Plugin system for custom workers
- [ ] MCP server discovery & auto-configuration
- [ ] Web dashboard (optional, separate package)
- [ ] PyPI publishing & release automation
- [ ] Documentation site

---

## Contributing

We welcome contributions! Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting a PR.

**Quick guidelines:**
1. Fork → Branch (`feat/your-feature`) → PR against `main`
2. All code must pass `ruff check`, `mypy --strict`, and `pytest`
3. Every PR must include tests for new functionality
4. Follow the constraints in [INSTRUCTIONS.md](INSTRUCTIONS.md) — they are non-negotiable

---

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <sub>Built with conviction that AI agents must be <strong>safe</strong>, <strong>observable</strong>, and <strong>under human control</strong>.</sub>
</p>
