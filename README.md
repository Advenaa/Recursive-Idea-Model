# The Recursive Idea Model (RIM)

A framework for challenging, decomposing, and synthesizing ideas through continual learning AI systems and agent swarm orchestration.

## Overview

RIM proposes an AI system that recursively decomposes, challenges, and synthesizes ideas — combining depth-first recursive analysis with breadth-first parallel orchestration. The framework draws on five key developments in AI research:

- **DSPy** (Stanford NLP) — Declarative AI programming and automatic optimization
- **Recursive Language Models** (Zhang et al.) — Managing unbounded analytical context
- **AOrchestra** (Ruan et al.) — Dynamic sub-agent creation
- **Kimi K2.5 Agent Swarm & PARL** (Moonshot AI) — Parallel agent orchestration via reinforcement learning
- **Continual Learning** (Bhalla, 2026) — Memory systems and feedback loops for persistent improvement

## Operating Mode

The project default is **Deep mode**.

- Deeper recursive analysis is the baseline (target depth: 3-5+ when feasible)
- Multi-pass challenge and synthesis are preferred over speed
- Longer end-to-end runtime is acceptable when it improves rigor
- Use fast mode only when explicitly requested

## Product Document

- `PRD.md` - Product Requirements Document for RIM MVP (Deep mode default)
- `TECH_SPEC.md` - Technical implementation spec (Codex CLI + Claude CLI execution)

## MVP Scaffold

The repository now includes a Python MVP scaffold under `rim/` with:

- FastAPI service (`rim/api/app.py`)
- Orchestrator pipeline (`rim/core/orchestrator.py`)
- Provider adapters for `codex` and `claude` CLIs (`rim/providers/`)
- SQLite persistence (`rim/storage/`)
- Benchmark runner + dataset (`rim/eval/`)
- Local CLI entrypoint (`rim/cli.py`)

Quickstart:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Python version policy:

- Required: Python `3.14.x`
- Recommended baseline: Python `3.14.3` (latest stable line as of February 14, 2026)

Run API:

```bash
uvicorn rim.api.app:app --reload
```

API usage (non-blocking by default):

```bash
curl -s -X POST "http://127.0.0.1:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"idea":"Build an AI CFO for freelancers","mode":"deep"}'

curl -s "http://127.0.0.1:8000/runs/<run_id>"
```

API usage (blocking for one call):

```bash
curl -s -X POST "http://127.0.0.1:8000/analyze?wait=true" \
  -H "Content-Type: application/json" \
  -d '{"idea":"Build an AI CFO for freelancers","mode":"deep"}'
```

Run CLI:

```bash
rim health
rim analyze --idea "Your idea here" --mode deep --json
rim eval run --mode deep --limit 3
```

Provider env vars:

```bash
export RIM_CODEX_CMD=codex
export RIM_CODEX_ARGS="exec --skip-git-repo-check --sandbox read-only"
export RIM_CLAUDE_CMD=claude
export RIM_CLAUDE_ARGS="-p --output-format json"
```

## Papers

### Main (SOTA)

| File | Description |
|------|-------------|
| `rim_paper_4.docx` | Current SOTA main paper |

### Archive

| File | Description |
|------|-------------|
| `archive/rbmo_paper_1.docx` | Original proposal — Recursive Business Model Optimization |
| `archive/rim_paper_1.docx` | V1 — Generalized to domain-agnostic idea challenging, expanded decomposition & synthesis |
| `archive/rim_paper_2.docx` | V2 — Integrated Kimi K2.5 Agent Swarm, PARL, Google scaling research, hybrid depth×breadth architecture |
| `archive/rim_paper_3.docx` | V3 draft kept for reference |

## Core Architecture

RIM consists of six layers working in recursive feedback loops:

1. **Decomposition** — Recursive engine breaking ideas into sub-components and assumptions
2. **Challenge** — Adversarial critique stress-testing each component
3. **Synthesis** — Integration engine rebuilding stronger ideas from critique results
4. **Orchestration** — Parallel agent swarm coordinating breadth at each recursive level
5. **Specialization** — Dynamic agent spawner creating domain-targeted analysts
6. **Learning** — Memory system + feedback loops for cross-session improvement

## Key Insight

> Neither depth nor breadth alone is sufficient. Depth without parallelism is slow and bottlenecked; parallelism without depth is broad but shallow. RIM combines both.

## License

All rights reserved.
