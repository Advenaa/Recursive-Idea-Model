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
- `TECH_SPEC.md` - Technical implementation spec (PI-first provider execution with Codex/Claude fallback)

## MVP Scaffold

The repository now includes a Python MVP scaffold under `rim/` with:

- FastAPI service (`rim/api/app.py`)
- Orchestrator adapter + reusable execution engine (`rim/core/orchestrator.py`, `rim/core/engine_runtime.py`)
- Multi-round disagreement arbitration with devil's-advocate + contract-aware specialist follow-up loops (`rim/agents/arbitrator.py`)
- Advanced verification adapters (`solver:`, `simulate:`, `data:`) (`rim/agents/advanced_verifier.py`)
- Scored specialist spawning with tool-routing contracts (`rim/agents/spawner.py`)
- Telemetry-driven specialist contract controller that auto-adjusts spawn role boosts from recent specialist arbitration outcomes (`rim/core/specialist_contract_quality.py`)
- Persistent API job queue (resume queued/running jobs on restart) (`rim/api/job_queue.py`)
- Provider adapters for `pi`, `codex`, and `claude` CLIs (`rim/providers/`)
- SQLite persistence + memory context reuse (`rim/storage/`)
- Benchmark runner + canonical 20-idea dataset (`rim/eval/`)
- Domain-weighted benchmark scoring + domain trend deltas (`rim/eval/runner.py`)
- Blind-review packet generator for report evaluation (`rim eval blindpack`)
- Deterministic baseline + real single-call LLM baselines + regression gate (`rim eval baseline`, `rim eval baseline-llm`, `rim eval gate`)
- Local CLI entrypoint (`rim/cli.py`)

## How Flow Works

The main runtime path is:

1. Input arrives from API (`/analyze`) or CLI (`rim analyze`).
2. `RimOrchestrator` creates/loads a run and delegates execution to `RimExecutionEngine`.
3. Engine stages execute in order:
   - decomposition (`rim/agents/decomposer.py`)
   - parallel critics (`rim/agents/critics.py`)
   - reconciliation/arbitration (`rim/agents/reconciliation.py`, `rim/agents/arbitrator.py`)
   - synthesis (`rim/agents/synthesizer.py`)
   - verification (`rim/agents/verification.py`, optional executable/advanced verification)
   - memory folding + persistence (`rim/core/memory_folding.py`, `rim/storage/repo.py`)
4. Each stage calls the provider router (`rim/providers/router.py`) for model output.
5. Router selects providers by stage policy (default PI-first), applies retry/repair logic, and enforces run budgets.
6. Run result, stage logs, telemetry, and errors are written to SQLite via `RunRepository`.

Provider routing behavior:

- Default order is `pi,codex,claude` (`RIM_PROVIDER_ORDER`).
- Set `RIM_PI_ONLY=1` for strict PI-only mode (no fallback to Codex/Claude).
- `rim health` shows provider availability and DB status.

Run states:

- `queued -> running -> completed`
- failure paths: `failed`, `partial`, `canceled`

Useful flow debug commands:

```bash
rim health
rim analyze --idea "Build an AI CFO for freelancers" --mode fast --json
rim run list --limit 5
rim run logs <run_id>
```

Embed in another project:

```python
import asyncio

from rim.engine import build_orchestrator
from rim.core.schemas import AnalyzeRequest

orchestrator = build_orchestrator()
request = AnalyzeRequest(idea="Your product idea", mode="deep")
result = asyncio.run(orchestrator.analyze(request))
print(result.synthesized_idea)
```

Custom agent packs for product-specific orchestration:

```python
from rim.engine import EngineAgentRegistry, build_engine

registry = EngineAgentRegistry()
custom_agents = registry.build(
    overrides={
        # replace any stage callable with your own implementation
        # "run_critics": my_custom_critics,
    }
)
engine = build_engine(agents=custom_agents)
```

Load packs from a config file without code edits:

```json
{
  "packs": {
    "my_pack": {
      "base_pack": "default",
      "overrides": {
        "run_critics": "myapp.rim_plugins:run_critics"
      }
    }
  }
}
```

```bash
export RIM_AGENT_PACKS_PATH=/absolute/path/agent_packs.json
export RIM_AGENT_PACK=my_pack
# API/CLI now resolve this pack automatically through rim.engine builders
```

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
rim eval baseline-llm --provider pi --mode deep --limit 3
rim eval baseline-llm --provider claude --mode deep --limit 3
rim eval baseline-llm --provider codex --mode deep --limit 3
rim eval duel --mode deep --limit 3 --baseline-provider pi --min-quality-delta 0.0
rim eval duel --mode deep --limit 3 --baseline-provider claude --min-quality-delta 0.0
rim eval list
rim eval compare
rim eval gate --min-quality-delta 0.0 --max-runtime-delta-sec 15
rim eval blindpack --limit 20
rim eval calibrate --target-quality 0.65 --target-runtime-sec 60
rim eval calibrate-loop --mode deep --limit 10 --target-quality 0.65 --target-runtime-sec 60
rim eval train-policy --reports-dir rim/eval/reports --target-quality 0.65 --target-runtime-sec 60
rim eval train-specialist-policy --reports-dir rim/eval/reports --target-quality 0.65 --target-runtime-sec 60
rim eval train-arbitration-policy --reports-dir rim/eval/reports --target-quality 0.65 --target-runtime-sec 60
rim eval train-spawn-policy --reports-dir rim/eval/reports --target-quality 0.65 --target-runtime-sec 60
rim eval train-memory-policy --reports-dir rim/eval/reports --target-quality 0.65 --target-runtime-sec 60
rim eval train-rl-policy --reports-dir rim/eval/reports --target-quality 0.65 --target-runtime-sec 60 --learning-rate 0.18 --epochs 3
rim eval train-rl-spawn-policy --reports-dir rim/eval/reports --target-quality 0.65 --target-runtime-sec 60 --learning-rate 0.18 --epochs 3
rim eval train-rl-memory-policy --reports-dir rim/eval/reports --target-quality 0.65 --target-runtime-sec 60 --learning-rate 0.18 --epochs 3
rim eval train-rl-orchestration-policy --reports-dir rim/eval/reports --target-quality 0.65 --target-runtime-sec 60 --learning-rate 0.18 --epochs 3
rim eval autolearn --mode deep --limit 10 --iterations 3 --lookback-reports 8 --optimizer rl --target-quality 0.65 --target-runtime-sec 60 --learning-rate 0.35 --rl-epochs 3
# autolearn updates depth/specialist/arbitration/spawn/memory policy files under rim/eval/policies by default
# specialist policy includes contract-controller defaults
# spawn policy can include specialist outcome-informed role boosts + dynamic token boosts + token/default routing/tool contracts
# memory policy can include memory-quality-controller defaults under RL autolearn
```

PI setup (required for PI-backed calls):

```bash
pi --list-models
# if not logged in:
pi
# then use /login and pick provider(s)

# optional pinning for RIM PI adapter:
export RIM_PI_PROVIDER=openai-codex
export RIM_PI_MODEL=gpt-5.1-codex-mini
```

Provider env vars:

```bash
export RIM_CODEX_CMD=codex
export RIM_CODEX_ARGS="exec --skip-git-repo-check --sandbox read-only"
export RIM_PI_CMD=pi
export RIM_PI_ARGS="--print --no-session --mode text"
export RIM_PROVIDER_ORDER="pi,codex,claude"
# Set to 1 to force strict PI-only runtime (no codex/claude fallback)
export RIM_PI_ONLY=0
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
# Optional trained depth policy file (used by autolearn loop)
export RIM_DEPTH_POLICY_PATH=rim/eval/policies/depth_policy.json
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
# Runtime long-horizon memory quality guardrails (auto-tighten fold params on degradation trends)
export RIM_ENABLE_MEMORY_QUALITY_CONTROLLER=1
export RIM_MEMORY_QUALITY_LOOKBACK_RUNS=24
export RIM_MEMORY_QUALITY_MIN_FOLDS=4
# Optional trained memory policy file from `rim eval train-memory-policy` or
# `rim eval train-rl-memory-policy` output. This policy can also carry
# memory-quality-controller defaults.
export RIM_MEMORY_POLICY_PATH=rim/eval/policies/memory_policy.json
export RIM_ENABLE_DISAGREEMENT_ARBITRATION=1
export RIM_ARBITRATION_MAX_JOBS=2
export RIM_ENABLE_DEVILS_ADVOCATE_ARBITRATION=1
export RIM_DEVILS_ADVOCATE_ROUNDS=1
export RIM_DEVILS_ADVOCATE_MIN_CONFIDENCE=0.72
# Optional trained arbitration policy file from `rim eval train-arbitration-policy` output
export RIM_ARBITRATION_POLICY_PATH=rim/eval/policies/arbitration_policy.json
export RIM_ENABLE_SPECIALIST_ARBITRATION_LOOP=1
export RIM_SPECIALIST_ARBITRATION_MAX_JOBS=2
export RIM_SPECIALIST_ARBITRATION_MIN_CONFIDENCE=0.78
# This policy can also carry specialist contract-controller defaults.
export RIM_SPECIALIST_POLICY_PATH=rim/eval/policies/specialist_policy.json
# Runtime specialist contract controller (auto-adjust spawn role boosts from recent specialist arbitration telemetry)
export RIM_ENABLE_SPECIALIST_CONTRACT_CONTROLLER=1
export RIM_SPECIALIST_CONTRACT_LOOKBACK_RUNS=24
export RIM_SPECIALIST_CONTRACT_MIN_ROUNDS=4
export RIM_SPECIALIST_CONTRACT_MIN_ROLE_SAMPLES=2
export RIM_SPAWN_MIN_ROLE_SCORE=1.0
export RIM_SPAWN_MAX_SPECIALISTS_DEEP=3
export RIM_SPAWN_MAX_SPECIALISTS_FAST=1
export RIM_ENABLE_DYNAMIC_SPECIALISTS=1
export RIM_SPAWN_MAX_DYNAMIC_SPECIALISTS=2
export RIM_SPAWN_ROLE_BOOSTS='{"security":0.4}'
export RIM_SPAWN_DYNAMIC_ROLE_CONTRACTS='{"bioinformatics":{"routing_policy":"prioritize_domain_specific_signals","tools":["context_probe:bioinformatics","evidence_scan"]}}'
export RIM_SPAWN_DYNAMIC_DEFAULT_CONTRACT='{"routing_policy":"prioritize_domain_specific_signals","tools":["evidence_scan","counterexample_search"]}'
export RIM_SPAWN_DYNAMIC_TOKEN_BOOSTS='{"bioinformatics":0.7}'
export RIM_SPAWN_ROLE_ROUTING_OVERRIDES='{"security":"prioritize_high_severity_and_compliance_constraints"}'
export RIM_SPAWN_ROLE_TOOL_OVERRIDES='{"security":["threat_model","policy_checklist"]}'
# Optional trained spawn policy file from `rim eval train-spawn-policy` output
export RIM_SPAWN_POLICY_PATH=rim/eval/policies/spawn_policy.json
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
export RIM_ADV_VERIFY_ALLOW_HTTP_DATA=0
export RIM_ADV_VERIFY_HTTP_TIMEOUT_SEC=5
export RIM_ADV_VERIFY_HTTP_MAX_BYTES=300000
# Optional comma-separated host allowlist for HTTP `data:` references
export RIM_ADV_VERIFY_HTTP_ALLOWED_HOSTS="example.com,docs.example.com"
# Built-in solver backend for `solver:` checks (`ast` or `z3`)
export RIM_ADV_VERIFY_SOLVER_BACKEND=ast
# `formal:` checks prefer z3 and can fall back to AST evaluation when enabled
export RIM_ADV_VERIFY_FORMAL_ALLOW_AST_FALLBACK=1
# Upper bound for count variables in formal symbolic proofs
export RIM_ADV_VERIFY_FORMAL_MAX_COUNT=200
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
- Prefix constraint with `solver:` to run deterministic symbolic assertions (`RIM_ADV_VERIFY_SOLVER_BACKEND=ast|z3`).
- Prefix constraint with `formal:` / `theorem:` / `constraint:` to force formal solver checks (z3-first with optional AST fallback).
  - Supports `| mode=prove|satisfiable|refute` (default `prove`).
  - Supports assumptions via `| assume=expr_a;expr_b`.
- Prefix constraint with `simulate:` for Monte Carlo robustness checks (`| trials=200 | min_pass_rate=0.7` supported).
- Prefix constraint with `data:` for data-reference checks (`| path=... | min_overlap=... | mode=all|fraction` supported).
  - Optional HTTP source: `| url=https://...` (requires `RIM_ADV_VERIFY_ALLOW_HTTP_DATA=1`; optional host allowlist via `RIM_ADV_VERIFY_HTTP_ALLOWED_HOSTS`).
- Available variables in expressions: `confidence_score`, `change_count`, `risk_count`, `experiment_count`, `finding_count`, `high_finding_count`, `critical_finding_count`.
- Example: `python: confidence_score >= 0.7 and risk_count <= 2`
- Example: `solver: confidence_score >= 0.75 and risk_count <= 2`
- Example: `formal: confidence_score >= 0.75 and risk_count <= 2`
- Example: `formal: risk_count <= finding_count | mode=prove | assume=finding_count >= 0;risk_count >= 0`
- Example: `formal: confidence_score >= 0.9 | mode=satisfiable | assume=confidence_score <= 1`
- Example: `simulate: confidence_score >= 0.65 | trials=300 | min_pass_rate=0.75`
- Example: `data: compliance, audit | path=rim/eval/data/benchmark_ideas.jsonl | min_overlap=0.5`
- Example: `data: compliance, audit | url=https://example.com/reference.txt | min_overlap=0.5`
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
