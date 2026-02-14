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
5. `RIM_CODEX_ENABLE_FEATURES` default `collab` (Codex experimental feature enabled by default)
6. `RIM_CODEX_DISABLE_FEATURES` optional comma/space-separated disable list
7. `RIM_PROVIDER_TIMEOUT_SEC` default `180`
8. `RIM_MAX_PARALLEL_CRITICS` default `6`
9. `RIM_RUN_MAX_PROVIDER_CALLS` default `120`
10. `RIM_RUN_MAX_PROVIDER_LATENCY_MS` default `900000`
11. `RIM_RUN_MAX_ESTIMATED_TOKENS` default `500000`
12. `RIM_RUN_MAX_ESTIMATED_COST_USD` default `10.0`
13. `RIM_PROVIDER_MAX_RETRIES` default `2` (transient provider retries before fallback)
14. `RIM_PROVIDER_RETRY_BASE_MS` default `250` (exponential backoff base delay)
15. `RIM_QUEUE_WORKERS` default `1`
16. `RIM_DETERMINISM_MODE` default `strict` (`off|strict|balanced`)
17. `RIM_DETERMINISM_SEED` default `42`
18. `RIM_JSON_REPAIR_RETRIES` default `1` (retry invalid JSON response once before provider fallback)
19. `RIM_MEMORY_MAX_AGE_DAYS` default `120`
20. `RIM_MEMORY_MIN_SEVERITY` default `medium` (or `low` for deep-mode override)
21. `RIM_MAX_ANALYSIS_CYCLES` default `1` (max recursive decompose-challenge-synthesize cycles per run)
22. `RIM_DEPTH_ALLOCATOR_MIN_CONFIDENCE` default `0.78` (confidence threshold for stopping recursion)
23. `RIM_DEPTH_ALLOCATOR_MAX_RESIDUAL_RISKS` default `2` (residual risk tolerance before recurse)
24. `RIM_DEPTH_ALLOCATOR_MAX_HIGH_FINDINGS` default `1` (high-severity finding tolerance before recurse)
25. `RIM_RECONCILE_CONSENSUS_MIN_AGENTS` default `3` (minimum distinct critic roles to mark consensus flaw)
26. `RIM_RECONCILE_CONSENSUS_MIN_CONFIDENCE` default `0.7` (minimum average confidence for consensus flaw)
27. `RIM_RECONCILE_MIN_UNIQUE_CRITICS` default `3` (minimum unique critic roles per node before diversity flag)
28. `RIM_RECONCILE_MAX_SINGLE_CRITIC_SHARE` default `0.7` (maximum share for one critic role before dominance flag)
29. `RIM_ENABLE_DOMAIN_CRITIC` default `1` (append one domain-specialist critic stage when domain is provided)
30. `RIM_SPAWN_MIN_ROLE_SCORE` default `1.0` (minimum specialist score before adding extra critic role)
31. `RIM_SPAWN_MAX_SPECIALISTS_DEEP` default `3` (max extra specialist critics in deep mode)
32. `RIM_SPAWN_MAX_SPECIALISTS_FAST` default `1` (max extra specialist critics in fast mode)
33. `RIM_ENABLE_DYNAMIC_SPECIALISTS` default `1` (enable runtime-generated dynamic specialist roles from unmatched high-signal tokens)
34. `RIM_SPAWN_MAX_DYNAMIC_SPECIALISTS` default `2` (max dynamic specialist candidates considered per run)
35. `RIM_SPAWN_POLICY_PATH` optional JSON policy file from `rim eval train-spawn-policy` (applies spawn defaults before env overrides)
36. `RIM_ENABLE_VERIFICATION` default `1` in deep mode, `0` in fast mode (deterministic post-synthesis checks)
37. `RIM_VERIFY_MIN_CONSTRAINT_OVERLAP` default `0.6` (minimum lexical overlap for constraint coverage)
38. `RIM_VERIFY_MIN_FINDING_OVERLAP` default `0.35` (minimum lexical overlap for high-risk finding coverage)
39. `RIM_ENABLE_MEMORY_FOLDING` default `1` in deep mode (fold cycle context into compact tripartite memory)
40. `RIM_MEMORY_FOLD_MAX_ENTRIES` default `12` (max folded context entries carried to next cycle)
41. `RIM_MEMORY_FOLD_NOVELTY_FLOOR` default `0.35` (minimum fraction of new fold entries before degradation flag)
42. `RIM_MEMORY_FOLD_MAX_DUPLICATE_RATIO` default `0.5` (max duplicate ratio before degradation flag)
43. `RIM_MEMORY_POLICY_PATH` optional JSON policy file from `rim eval train-memory-policy` (applies memory-fold defaults before env overrides)
44. `RIM_ENABLE_DISAGREEMENT_ARBITRATION` default `1` in deep mode (resolve critique disagreements before synthesis)
45. `RIM_ARBITRATION_MAX_JOBS` default `2` (max disagreement arbitration calls per cycle)
46. `RIM_ENABLE_DEVILS_ADVOCATE_ARBITRATION` default `1` in deep mode (run follow-up devil's-advocate arbitration on low-confidence/escalated decisions)
47. `RIM_DEVILS_ADVOCATE_ROUNDS` default `1` (max devil's-advocate rounds per cycle)
48. `RIM_DEVILS_ADVOCATE_MIN_CONFIDENCE` default `0.72` (minimum arbitration confidence before triggering devil follow-up)
49. `RIM_ENABLE_SPECIALIST_ARBITRATION_LOOP` default `1` in deep mode (run specialist arbitration on diversity-flagged disagreements)
50. `RIM_SPECIALIST_ARBITRATION_MAX_JOBS` default `2` (max specialist arbitration reviews per cycle)
51. `RIM_SPECIALIST_ARBITRATION_MIN_CONFIDENCE` default `0.78` (minimum arbitration confidence before specialist follow-up)
52. `RIM_SPECIALIST_POLICY_PATH` optional JSON policy file from `rim eval train-specialist-policy` (applies specialist arbitration defaults before env overrides)
53. `RIM_ENABLE_EXECUTABLE_VERIFICATION` default `1` in deep mode (run safe executable checks from prefixed constraints)
54. `RIM_EXEC_VERIFY_MAX_CHECKS` default `5` (max executable constraint checks per cycle)
55. `RIM_ENABLE_PYTHON_EXEC_CHECKS` default `0` (allow `python_exec:` subprocess checks)
56. `RIM_PYTHON_EXEC_TIMEOUT_SEC` default `2` (timeout per `python_exec:` check)
57. `RIM_ENABLE_ADVANCED_VERIFICATION` default `1` in deep mode (run solver/simulation/data-backed verification)
58. `RIM_ADV_VERIFY_MAX_CHECKS` default `4` (max advanced verification checks per cycle)
59. `RIM_ADV_VERIFY_SIMULATION_TRIALS` default `200` (Monte Carlo samples per simulation check)
60. `RIM_ADV_VERIFY_SIMULATION_MIN_PASS_RATE` default `0.7` (minimum simulation pass-rate threshold)
61. `RIM_ADV_VERIFY_SIMULATION_SEED` default `42` (deterministic simulation seed)
62. `RIM_ADV_VERIFY_DATA_PATH` default `rim/eval/data/benchmark_ideas.jsonl` (local reference dataset path for `data:` checks)
63. `RIM_ADV_VERIFY_EXTERNAL_TIMEOUT_SEC` default `8` (timeout for optional external advanced-verification adapters)
64. `RIM_ADV_VERIFY_EXTERNAL_SOLVER_CMD` optional command (external backend for `solver:` checks)
65. `RIM_ADV_VERIFY_EXTERNAL_SIMULATION_CMD` optional command (external backend for `simulate:` checks)
66. `RIM_ADV_VERIFY_EXTERNAL_DATA_CMD` optional command (external backend for `data:` checks)
67. Reference adapter template: `scripts/advanced_verify_adapter.py` (stdin JSON -> stdout JSON contract)
68. Optional adapter env: `RIM_ADV_VERIFY_ADAPTER_SOLVER_BACKEND=z3` (use z3 backend when available, fallback otherwise)
69. `RIM_DEPTH_POLICY_PATH` optional JSON policy file (recommended output from `rim eval autolearn`) that applies depth-allocator defaults before env overrides

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
5. `domain` (text nullable)
6. `severity` (text: low|medium|high|critical)
7. `score` (real)
8. `created_at` (datetime)

`run_feedback`

1. `id` (text primary key)
2. `run_id` (fk runs.id)
3. `verdict` (text: accept|reject)
4. `notes` (text nullable)
5. `created_at` (datetime)

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
2. Supports idempotent `run_id` query parameter:
   - same `run_id` + same request returns existing run state,
   - same `run_id` + different request returns HTTP 409.
3. Persists request payload in DB and enqueues run in persistent worker queue.
4. Starts pipeline execution in background by default.
5. Returns HTTP 202 for newly created queued runs; returns HTTP 200 when reusing existing `run_id`.
6. When `wait=true`, blocks until completion and returns full run payload.
7. On stage failure, response includes structured error:
   - `stage`,
   - `provider`,
   - `message`,
   - `retryable`.

### 7.2 `GET /runs`

Returns recent runs with optional filters:

1. `status` (`queued|running|completed|failed|partial|canceled`)
2. `mode` (`deep|fast`)
3. `limit` (default `20`, max `200`)
4. `offset` (default `0`)

### 7.3 `GET /runs/{run_id}`

Returns persisted run state and final output.

### 7.4 `GET /health`

Returns:

1. service health,
2. database health,
3. `codex` adapter health,
4. `claude` adapter health.

### 7.5 `GET /runs/{run_id}/logs`

Returns ordered stage telemetry for the run:

1. stage,
2. provider,
3. latency,
4. status,
5. stage metadata (including decomposition stop reason when applicable).

### 7.6 `POST /runs/{run_id}/cancel`

Behavior:

1. Cancels a queued/running run.
2. Marks run status as `canceled` with structured queue-stage error metadata.
3. If run is already terminal, returns current run state without mutation.

### 7.7 `POST /runs/{run_id}/retry`

Behavior:

1. Retry is allowed only for `failed|partial|canceled` runs.
2. Clears prior run artifacts (nodes/findings/synthesis/memory/feedback/logs) and re-queues the run.
3. Returns HTTP 202 with `status: queued` when accepted.
4. Returns HTTP 409 for non-retryable statuses.

### 7.8 `POST /runs/{run_id}/feedback`

Request JSON:

```json
{
  "verdict": "accept|reject",
  "notes": "optional"
}
```

Behavior:

1. Stores run-level feedback.
2. Re-scores memory entries derived from that run.
3. Optionally persists feedback notes as memory for future retrieval.

## 8) Local CLI Specification

### 8.1 Analyze Command

`rim analyze --idea "..." --mode deep`

Flags:

1. `--mode deep|fast` (default `deep`)
2. `--domain <text>`
3. `--constraint <text>` (repeatable)
4. `--run-id <text>` (idempotent run key)
5. `--json` (print raw JSON)
6. `--save <path>` (write output artifact)

### 8.2 Run Inspection and Control

`rim run show <run_id>`

`rim run list --limit 20 --status completed`

`rim run cancel <run_id>`

`rim run retry <run_id>`

### 8.3 Healthcheck

`rim health`

### 8.4 Benchmark

`rim eval run --mode deep --limit 10 --save report.json`

Additional eval commands:

1. `rim eval list` for saved report history.
2. `rim eval compare --base <report_a> --target <report_b>` for time-over-time deltas (defaults to latest two reports when omitted).
3. `rim eval baseline --limit 10 --save baseline.json` for deterministic single-pass baseline outputs.
4. `rim eval gate --base <report_a> --target <report_b> --min-quality-delta 0.0 --max-runtime-delta-sec 15` for pass/fail regression checks.
5. `rim eval duel --mode deep --limit 10 --min-quality-delta 0.0` to run baseline + RIM benchmark and gate in one step.
6. `rim eval blindpack --report <report.json> --limit 20 --save blind_review.json` to generate anonymized review packets.
7. `rim eval calibrate --report <report.json> --target-quality 0.65 --target-runtime-sec 60` to recommend depth-allocator env settings from benchmark signals.
8. `rim eval calibrate-loop --mode deep --limit 10 --target-quality 0.65 --target-runtime-sec 60` to run benchmark + calibration in one step.
9. `rim eval train-policy --reports-dir rim/eval/reports --target-quality 0.65 --target-runtime-sec 60` to aggregate multiple reports into a depth-policy recommendation.
10. `rim eval train-specialist-policy --reports-dir rim/eval/reports --target-quality 0.65 --target-runtime-sec 60` to aggregate benchmark telemetry into specialist-arbitration policy defaults.
11. `rim eval train-spawn-policy --reports-dir rim/eval/reports --target-quality 0.65 --target-runtime-sec 60` to aggregate benchmark telemetry into specialization spawn-policy defaults.
12. `rim eval train-memory-policy --reports-dir rim/eval/reports --target-quality 0.65 --target-runtime-sec 60` to aggregate benchmark telemetry into memory-fold policy defaults.
13. `rim eval autolearn --mode deep --limit 10 --iterations 3 --lookback-reports 8 --target-quality 0.65 --target-runtime-sec 60 --learning-rate 0.35` to run benchmark cycles and auto-update depth + specialist policies from fresh telemetry.

## 9) Orchestration Logic

### 9.1 Stage Pipeline

1. `intake`
2. `memory_read`
3. `specialization_spawn` (build per-run specialist critic plan)
4. `decompose` (cycle N)
5. `challenge_parallel` (cycle N)
6. `challenge_reconciliation` (consensus/disagreement aggregation)
7. `challenge_arbitration` (optional disagreement resolution calls, including devil's-advocate and specialist follow-up rounds)
8. `synthesis` (cycle N)
9. `verification` (deterministic constraint/risk coverage checks)
10. `verification_executable` (safe expression checks for prefixed executable constraints)
11. `verification_advanced` (solver/simulation/data-reference checks)
12. `depth_allocator` (decide recurse or stop)
13. `memory_fold` (when recursing; compact episodic/working/tool summaries)
14. Repeat steps 4-13 while recursion decision is true and cycle budget remains
15. `memory_write`
16. `provider_budget`
17. `finalize`

### 9.2 Parallel Challenge

1. Build critic job list for each active node.
2. Execute jobs concurrently with semaphore limit.
3. Validate each critic output against schema.
4. Retry invalid responses once with stricter prompt.
5. Persist findings incrementally.

### 9.3 Executable Verification

1. Constraints prefixed with `python:`, `py:`, or `assert:` are treated as executable checks.
2. Constraints prefixed with `python_exec:` are treated as timed subprocess checks when explicitly enabled.
3. Expression checks are evaluated with a safe evaluator (no function calls, attribute access, or imports).
4. `python_exec:` checks run with a restricted builtins set and must set `passed = True|False`.
5. Failed executable checks are logged and converted into residual risks with confidence penalty.
6. Supported context variables:
   - `confidence_score`
   - `change_count`
   - `risk_count`
   - `experiment_count`
   - `finding_count`
   - `high_finding_count`
   - `critical_finding_count`

### 9.4 Specialization Spawn

1. `specialization_spawn` builds a per-run specialist critic plan from domain, constraints, and recent memory context.
2. The spawner selects extra critic roles via keyword-to-role rules (security, finance, scalability, UX).
3. Deep mode can include up to three extra specialists; fast mode caps at one.
4. Selected specialists are appended to the challenge layer as additional critic stages.

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
6. Run artifacts are persisted and retrievable via `GET /runs` and `GET /runs/{run_id}`.
7. Re-submitting `POST /analyze` with same `run_id` is idempotent and does not create duplicate runs.

## 17) Self-Iteration Workflow

Local script: `scripts/self_iteration.sh`

Phases:

1. `prepare`:
   - generate proposal via `rim analyze`,
   - run compile + tests,
   - store baseline eval report.
2. `verify`:
   - rerun compile + tests after edits,
   - run eval report and regression gate against baseline,
   - fail non-zero on gate failure.

Guardrail policy: `docs/SELF_ITERATION_POLICY.md`
