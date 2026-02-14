# Technical Specification (TECH SPEC)

## Product

Recursive Idea Model (RIM) MVP

## Version

v0.1 (February 14, 2026)

## 1) Scope

This document translates `PRD.md` into an implementable system design.

MVP scope:

1. End-to-end pipeline: intake -> decomposition -> challenge -> synthesis -> memory.
2. Default Deep mode.
3. Local orchestration using **Codex CLI** and **Claude CLI** as model execution engines.
4. Developer-facing interfaces: HTTP API and local CLI.

## 2) Core Technical Decisions

1. Language: Python 3.14.x (recommended baseline: 3.14.3).
2. API framework: FastAPI.
3. Storage: SQLite for MVP.
4. Concurrency: `asyncio` + process-level parallelism for critic calls.
5. LLM backend strategy: local shell adapters for:
   - `codex` CLI
   - `claude` CLI
6. Default runtime profile: Deep mode (depth 3-5, multi-pass critique).

## 3) System Architecture

### 3.1 Components

1. `api`:
   - HTTP endpoints for run creation and retrieval.
2. `orchestrator`:
   - controls recursion, stage sequencing, branch budgets, retries.
3. `agents`:
   - decomposition, critic, synthesis role definitions.
4. `provider_router`:
   - routes stage calls to `codex` or `claude` CLI per policy.
5. `persistence`:
   - SQLite models and repository access.
6. `evaluation`:
   - benchmark harness and quality scoring.
7. `telemetry`:
   - logs, latency, token usage, stage-level outcomes.

### 3.2 High-Level Flow

1. User submits idea via API/CLI.
2. Orchestrator creates `run_id` and persists initial run state.
3. Decomposer recursively expands idea tree to target depth.
4. Critic workers run in parallel on each node.
5. Synthesizer merges findings into revised idea and structured outputs.
6. Memory module stores artifacts and indexed summaries.
7. API returns output contract JSON.

## 4) Provider Strategy (Codex CLI + Claude CLI)

### 4.1 Rationale

You are on max subscriptions, so MVP should directly leverage both CLI products instead of paying for additional hosted APIs.

### 4.2 Routing Policy

Default policy:

1. Decomposition: `codex` (strong structured decomposition behavior).
2. Critique: split across both:
   - logic + execution critics on `codex`,
   - evidence + adversarial critic on `claude`.
3. Synthesis: dual-pass:
   - first draft on `claude`,
   - refinement/constraint check on `codex`.

Fallback policy:

1. If chosen provider fails, retry same stage once on alternate provider.
2. If both fail, return partial output with explicit stage error.

### 4.3 CLI Adapter Contract

Each provider adapter must implement:

1. `invoke(prompt: str, config: ProviderConfig) -> ProviderResult`
2. `invoke_json(prompt: str, config: ProviderConfig, json_schema?: dict) -> dict`
3. `healthcheck() -> bool`

`ProviderResult` fields:

1. `text`
2. `raw_output`
3. `latency_ms`
4. `estimated_tokens_in`
5. `estimated_tokens_out`
6. `provider` (`codex|claude`)
7. `exit_code`

### 4.4 Execution Notes

Adapters run shell commands via subprocess with timeout and stderr capture.

Environment variables:

1. `RIM_CODEX_CMD` default `codex`
2. `RIM_CLAUDE_CMD` default `claude`
3. `RIM_CODEX_ARGS` default `exec --skip-git-repo-check --sandbox read-only`
4. `RIM_CLAUDE_ARGS` default `-p --output-format json`
5. `RIM_PROVIDER_TIMEOUT_SEC` default `180`
6. `RIM_MAX_PARALLEL_CRITICS` default `6`

## 5) Modes and Runtime Controls

### 5.1 Deep Mode (Default)

1. `max_depth = 4` (configurable to 5)
2. `critics_per_node = 4`
3. `synthesis_passes = 2`
4. `self_critique_pass = enabled`
5. `evidence_requirement = strict`

### 5.2 Fast Mode (Explicit only)

1. `max_depth = 2`
2. `critics_per_node = 2`
3. `synthesis_passes = 1`
4. `self_critique_pass = disabled`

### 5.3 Stop Conditions

1. Depth limit reached.
2. Node confidence above threshold.
3. Marginal gain below threshold for two consecutive levels.
4. Global runtime budget exceeded.

## 6) Data Model (SQLite)

### 6.1 Tables

`runs`

1. `id` (text primary key)
2. `mode` (text)
3. `input_idea` (text)
4. `status` (text: running|completed|failed|partial)
5. `created_at` (datetime)
6. `completed_at` (datetime nullable)
7. `confidence_score` (real nullable)
8. `error_summary` (text nullable)

`nodes`

1. `id` (text primary key)
2. `run_id` (fk runs.id)
3. `parent_node_id` (text nullable)
4. `depth` (integer)
5. `component_text` (text)
6. `node_type` (text: claim|assumption|dependency)
7. `confidence` (real nullable)

`critic_findings`

1. `id` (text primary key)
2. `run_id` (fk runs.id)
3. `node_id` (fk nodes.id)
4. `critic_type` (text)
5. `issue` (text)
6. `severity` (text: low|medium|high|critical)
7. `confidence` (real)
8. `suggested_fix` (text)
9. `provider` (text)

`synthesis_outputs`

1. `id` (text primary key)
2. `run_id` (fk runs.id)
3. `synthesized_idea` (text)
4. `changes_summary_json` (text)
5. `residual_risks_json` (text)
6. `next_experiments_json` (text)

`memory_entries`

1. `id` (text primary key)
2. `run_id` (fk runs.id)
3. `entry_type` (text: pattern|failure|heuristic|insight)
4. `entry_text` (text)
5. `score` (real)
6. `created_at` (datetime)

`stage_logs`

1. `id` (text primary key)
2. `run_id` (fk runs.id)
3. `stage` (text)
4. `provider` (text nullable)
5. `latency_ms` (integer)
6. `status` (text)
7. `meta_json` (text)
8. `created_at` (datetime)

## 7) API Specification

### 7.1 `POST /analyze`

Request JSON:

```json
{
  "idea": "string",
  "mode": "deep",
  "domain": "optional",
  "constraints": ["optional"],
  "desired_outcome": "optional"
}
```

Behavior:

1. Creates a run.
2. Starts pipeline execution in background by default.
3. Returns `{ run_id, status }` with HTTP 202 when `wait=false`.
4. When `wait=true`, blocks until completion and returns full run payload.

### 7.2 `GET /runs/{run_id}`

Returns persisted run state and final output.

### 7.3 `GET /health`

Returns:

1. service health,
2. database health,
3. `codex` adapter health,
4. `claude` adapter health.

## 8) Local CLI Specification

### 8.1 Analyze Command

`rim analyze --idea "..." --mode deep`

Flags:

1. `--mode deep|fast` (default `deep`)
2. `--domain <text>`
3. `--constraint <text>` (repeatable)
4. `--json` (print raw JSON)
5. `--save <path>` (write output artifact)

### 8.2 Run Inspection

`rim run show <run_id>`

### 8.3 Healthcheck

`rim health`

### 8.4 Benchmark

`rim eval run --mode deep --limit 10 --save report.json`

## 9) Orchestration Logic

### 9.1 Stage Pipeline

1. `intake`
2. `decompose`
3. `challenge_parallel`
4. `synthesize_pass_1`
5. `self_critique` (deep only)
6. `synthesize_pass_2` (deep only)
7. `memory_write`
8. `finalize`

### 9.2 Parallel Challenge

1. Build critic job list for each active node.
2. Execute jobs concurrently with semaphore limit.
3. Validate each critic output against schema.
4. Retry invalid responses once with stricter prompt.
5. Persist findings incrementally.

## 10) Prompt and Schema Discipline

1. All stages must request strict JSON output.
2. Validate with Pydantic models.
3. On validation failure:
   - repair attempt by provider,
   - then fallback provider,
   - then partial-stage failure.

## 11) Reliability and Error Handling

1. Provider call timeout per stage.
2. Exponential backoff for transient failures.
3. Idempotent rerun support by `run_id`.
4. Partial completion support with explicit `status = partial`.
5. Structured error object in response:
   - `stage`,
   - `provider`,
   - `message`,
   - `retryable`.

## 12) Observability

1. Structured JSON logs.
2. Per-stage metrics:
   - latency,
   - retries,
   - provider selected,
   - parse/validation failures.
3. Run summary metrics:
   - total runtime,
   - node count,
   - critic finding count.

## 13) Security and Secrets

1. No provider keys committed to repo.
2. Use local CLI auth sessions for `codex` and `claude`.
3. Sanitize logs to avoid leaking sensitive idea text in shared environments.
4. Optional redaction mode for persisted memory fields.

## 14) Suggested Repository Layout

```text
RIM/
  README.md
  PRD.md
  TECH_SPEC.md
  rim/
    api/
      app.py
      models.py
    core/
      orchestrator.py
      modes.py
      schemas.py
    agents/
      decomposer.py
      critics.py
      synthesizer.py
    providers/
      base.py
      codex_cli.py
      claude_cli.py
      router.py
    storage/
      db.py
      models.py
      repo.py
    eval/
      benchmark.py
      scoring.py
    cli.py
  tests/
    test_orchestrator.py
    test_providers.py
    test_api.py
```

## 15) Delivery Plan

1. Sprint 1:
   - scaffold project,
   - implement provider adapters,
   - add health endpoint.
2. Sprint 2:
   - implement decomposition and challenge stages,
   - persist runs/nodes/findings.
3. Sprint 3:
   - implement synthesis and memory write,
   - ship CLI and API parity.
4. Sprint 4:
   - add benchmark harness,
   - tune deep mode defaults,
   - harden retries and telemetry.

## 16) Acceptance Criteria

1. `POST /analyze` starts non-blocking run creation and returns `run_id`.
2. `POST /analyze?wait=true` returns completed run payload for valid inputs.
3. Deep mode is default in both API and CLI paths.
4. Both `codex` and `claude` CLI adapters are implemented and health-checkable.
5. Failure of one provider can fall back to the other for at least one retry.
6. Run artifacts are persisted and retrievable via `GET /runs/{run_id}`.
