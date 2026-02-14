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
- Orchestrator pipeline with recursive stop conditions (`rim/core/orchestrator.py`)
- Multi-round disagreement arbitration with devil's-advocate + specialist follow-up loops (`rim/agents/arbitrator.py`)
- Advanced verification adapters (`solver:`, `simulate:`, `data:`) (`rim/agents/advanced_verifier.py`)
- Scored specialist spawning with tool-routing contracts (`rim/agents/spawner.py`)
- Persistent API job queue (resume queued/running jobs on restart) (`rim/api/job_queue.py`)
- Provider adapters for `codex` and `claude` CLIs (`rim/providers/`)
- SQLite persistence + memory context reuse (`rim/storage/`)
- Benchmark runner + canonical 20-idea dataset (`rim/eval/`)
- Domain-weighted benchmark scoring + domain trend deltas (`rim/eval/runner.py`)
- Blind-review packet generator for report evaluation (`rim eval blindpack`)
- Deterministic single-pass baseline + regression gate (`rim eval baseline`, `rim eval gate`)
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

CI:

- GitHub Actions workflow at `.github/workflows/ci.yml` runs compile + test on push/PR.

Run API:

```bash
uvicorn rim.api.app:app --reload
```

API usage (non-blocking by default):

```bash
curl -s -X POST "http://127.0.0.1:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"idea":"Build an AI CFO for freelancers","mode":"deep"}'

# Idempotent run creation/retry by run_id
curl -s -X POST "http://127.0.0.1:8000/analyze?run_id=demo-run-1" \
  -H "Content-Type: application/json" \
  -d '{"idea":"Build an AI CFO for freelancers","mode":"deep"}'

curl -s "http://127.0.0.1:8000/runs?limit=10&status=completed"
curl -s "http://127.0.0.1:8000/runs/<run_id>"
curl -s "http://127.0.0.1:8000/runs/<run_id>/logs"
curl -s -X POST "http://127.0.0.1:8000/runs/<run_id>/cancel"
curl -s -X POST "http://127.0.0.1:8000/runs/<run_id>/retry"
curl -s -X POST "http://127.0.0.1:8000/runs/<run_id>/feedback" \
  -H "Content-Type: application/json" \
  -d '{"verdict":"accept","notes":"Strong output"}'
```

API usage (blocking for one call):

```bash
curl -s -X POST "http://127.0.0.1:8000/analyze?wait=true" \
  -H "Content-Type: application/json" \
  -d '{"idea":"Build an AI CFO for freelancers","mode":"deep"}'
```

Failure contract:

- Run responses can return `status: "failed"`, `status: "partial"`, or `status: "canceled"`.
- Error payload is structured as `{ stage, provider, message, retryable }`.

Run CLI:

```bash
rim health
rim analyze --idea "Your idea here" --mode deep --json
rim analyze --idea "Your idea here" --mode deep --run-id demo-run-1 --json
rim run list --limit 10 --status completed
rim run cancel <run_id>
rim run retry <run_id>
rim run logs <run_id>
rim run feedback <run_id> --verdict accept --notes "Strong output"
rim eval run --mode deep --limit 3
rim eval baseline --limit 3
rim eval duel --mode deep --limit 3 --min-quality-delta 0.0
rim eval list
rim eval compare
rim eval gate --min-quality-delta 0.0 --max-runtime-delta-sec 15
rim eval blindpack --limit 20
rim eval calibrate --target-quality 0.65 --target-runtime-sec 60
rim eval calibrate-loop --mode deep --limit 10 --target-quality 0.65 --target-runtime-sec 60
rim eval train-policy --reports-dir rim/eval/reports --target-quality 0.65 --target-runtime-sec 60
rim eval train-specialist-policy --reports-dir rim/eval/reports --target-quality 0.65 --target-runtime-sec 60
```

Provider env vars:

```bash
export RIM_CODEX_CMD=codex
export RIM_CODEX_ARGS="exec --skip-git-repo-check --sandbox read-only"
# Experimental Codex features (enabled by default: collab)
export RIM_CODEX_ENABLE_FEATURES="collab"
# Optional:
# export RIM_CODEX_DISABLE_FEATURES="web_search_cached"
export RIM_CLAUDE_CMD=claude
export RIM_CLAUDE_ARGS="-p --output-format json"
export RIM_RUN_MAX_PROVIDER_CALLS=120
export RIM_RUN_MAX_PROVIDER_LATENCY_MS=900000
export RIM_RUN_MAX_ESTIMATED_TOKENS=500000
export RIM_RUN_MAX_ESTIMATED_COST_USD=10.0
export RIM_PROVIDER_MAX_RETRIES=2
export RIM_PROVIDER_RETRY_BASE_MS=250
export RIM_DETERMINISM_MODE=strict
export RIM_DETERMINISM_SEED=42
export RIM_JSON_REPAIR_RETRIES=1
export RIM_MEMORY_MAX_AGE_DAYS=120
export RIM_MEMORY_MIN_SEVERITY=medium
# Recursive cycle controls (default stays single-cycle for compatibility)
export RIM_MAX_ANALYSIS_CYCLES=1
export RIM_DEPTH_ALLOCATOR_MIN_CONFIDENCE=0.78
export RIM_DEPTH_ALLOCATOR_MAX_RESIDUAL_RISKS=2
export RIM_DEPTH_ALLOCATOR_MAX_HIGH_FINDINGS=1
export RIM_RECONCILE_CONSENSUS_MIN_AGENTS=3
export RIM_RECONCILE_CONSENSUS_MIN_CONFIDENCE=0.7
export RIM_RECONCILE_MIN_UNIQUE_CRITICS=3
export RIM_RECONCILE_MAX_SINGLE_CRITIC_SHARE=0.7
export RIM_ENABLE_DOMAIN_CRITIC=1
export RIM_ENABLE_VERIFICATION=1
export RIM_VERIFY_MIN_CONSTRAINT_OVERLAP=0.6
export RIM_VERIFY_MIN_FINDING_OVERLAP=0.35
export RIM_ENABLE_MEMORY_FOLDING=1
export RIM_MEMORY_FOLD_MAX_ENTRIES=12
export RIM_MEMORY_FOLD_NOVELTY_FLOOR=0.35
export RIM_MEMORY_FOLD_MAX_DUPLICATE_RATIO=0.5
export RIM_ENABLE_DISAGREEMENT_ARBITRATION=1
export RIM_ARBITRATION_MAX_JOBS=2
export RIM_ENABLE_DEVILS_ADVOCATE_ARBITRATION=1
export RIM_DEVILS_ADVOCATE_ROUNDS=1
export RIM_DEVILS_ADVOCATE_MIN_CONFIDENCE=0.72
export RIM_ENABLE_SPECIALIST_ARBITRATION_LOOP=1
export RIM_SPECIALIST_ARBITRATION_MAX_JOBS=2
export RIM_SPECIALIST_ARBITRATION_MIN_CONFIDENCE=0.78
# Optional trained specialist policy file from `rim eval train-specialist-policy` output
export RIM_SPECIALIST_POLICY_PATH=rim/eval/reports/specialist_policy.json
export RIM_SPAWN_MIN_ROLE_SCORE=1.0
export RIM_SPAWN_MAX_SPECIALISTS_DEEP=3
export RIM_SPAWN_MAX_SPECIALISTS_FAST=1
export RIM_ENABLE_DYNAMIC_SPECIALISTS=1
export RIM_SPAWN_MAX_DYNAMIC_SPECIALISTS=2
export RIM_ENABLE_EXECUTABLE_VERIFICATION=1
export RIM_EXEC_VERIFY_MAX_CHECKS=5
export RIM_ENABLE_PYTHON_EXEC_CHECKS=0
export RIM_PYTHON_EXEC_TIMEOUT_SEC=2
export RIM_ENABLE_ADVANCED_VERIFICATION=1
export RIM_ADV_VERIFY_MAX_CHECKS=4
export RIM_ADV_VERIFY_SIMULATION_TRIALS=200
export RIM_ADV_VERIFY_SIMULATION_MIN_PASS_RATE=0.7
export RIM_ADV_VERIFY_SIMULATION_SEED=42
export RIM_ADV_VERIFY_DATA_PATH=rim/eval/data/benchmark_ideas.jsonl
export RIM_ADV_VERIFY_EXTERNAL_TIMEOUT_SEC=8
# Optional external adapters (stdin JSON in, stdout JSON out)
export RIM_ADV_VERIFY_EXTERNAL_SOLVER_CMD="python scripts/advanced_verify_adapter.py"
export RIM_ADV_VERIFY_EXTERNAL_SIMULATION_CMD="python scripts/advanced_verify_adapter.py"
export RIM_ADV_VERIFY_EXTERNAL_DATA_CMD="python scripts/advanced_verify_adapter.py"
# Optional for adapter script:
# export RIM_ADV_VERIFY_ADAPTER_SOLVER_BACKEND=z3
```

Verification constraint formats:

- Prefix constraint with `python:` / `py:` / `assert:` to run a safe expression check.
- Prefix constraint with `python_exec:` to run an explicit Python snippet in a timed subprocess.
- Prefix constraint with `solver:` to run deterministic symbolic assertions.
- Prefix constraint with `simulate:` for Monte Carlo robustness checks (`| trials=200 | min_pass_rate=0.7` supported).
- Prefix constraint with `data:` for data-reference checks (`| path=... | min_overlap=... | mode=all|fraction` supported).
- Available variables in expressions: `confidence_score`, `change_count`, `risk_count`, `experiment_count`, `finding_count`, `high_finding_count`, `critical_finding_count`.
- Example: `python: confidence_score >= 0.7 and risk_count <= 2`
- Example: `solver: confidence_score >= 0.75 and risk_count <= 2`
- Example: `simulate: confidence_score >= 0.65 | trials=300 | min_pass_rate=0.75`
- Example: `data: compliance, audit | path=rim/eval/data/benchmark_ideas.jsonl | min_overlap=0.5`
- `python_exec:` snippets receive `context` and should set `passed = True|False` (optional `detail` string).
- External adapter command contract: read one JSON object from stdin with `{check_type,payload,context,synthesis}` and print JSON `{passed: bool, result?: any, error?: string}`.

Self-iteration loop:

```bash
scripts/self_iteration.sh prepare --objective "Improve queue reliability and logs"
# apply code changes for this objective
scripts/self_iteration.sh verify --iteration-dir artifacts/self-iterations/<iteration_dir>
```

Policy and guardrails: `docs/SELF_ITERATION_POLICY.md`

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
