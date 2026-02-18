# Product Requirements Document (PRD)

## Product

Recursive Idea Model (RIM) MVP

## Version

v0.2-draft (updated February 18, 2026)

## 1) Purpose

Build an MVP that turns a raw idea into a stronger, stress-tested version through recursive decomposition, adversarial challenge, and synthesis.

The MVP defaults to **Deep mode** (thorough analysis first, speed second).

## 2) Problem Statement

Most AI workflows produce one-pass answers that are broad but shallow. Users need a system that can:

- recursively break ideas into assumptions and components,
- challenge weaknesses from multiple angles,
- synthesize improved versions with explicit reasoning,
- and learn from prior runs.

## 3) Goals

1. Provide end-to-end recursive idea analysis in one run.
2. Make Deep mode the default operating behavior.
3. Produce outputs with clear "before -> after" improvement.
4. Persist run memory to improve future analyses.
5. Ship a usable developer-facing interface (CLI or API-first).
6. Replace bespoke provider execution with a PI-first runtime while preserving RIM analysis semantics and output contracts.

## 4) Non-Goals (MVP)

1. Full autonomous long-horizon agent economy.
2. Fine-tuning custom foundation models.
3. Production-grade multi-tenant SaaS.
4. Complex UI/UX beyond basic interaction surface.

## 5) Primary Users

1. Founder/operator testing business and strategy ideas.
2. Researcher stress-testing hypotheses.
3. Product/engineering lead evaluating solution concepts.

## 6) Core User Story

As a user, I submit an idea and receive a deeply analyzed revision with:

- recursive breakdown,
- multi-agent critique,
- synthesized improved version,
- residual risks,
- and recommended next experiments.

## 7) Product Principles

1. Depth over latency by default.
2. Explicit assumptions and evidence checks.
3. Parallel critique for breadth at each recursion layer.
4. Structured outputs over free-form text.
5. Traceable reasoning with run logs.

## 8) Functional Requirements

### FR-1 Idea Intake

- Accept idea input as free text.
- Optional metadata: domain, constraints, desired outcome.

### FR-2 Recursive Decomposition

- Decompose into components, claims, assumptions, dependencies.
- Default target recursion depth: 3-5 (configurable).
- Stop conditions:
  - max depth reached,
  - component confidence threshold reached,
  - marginal gain below threshold.

### FR-3 Parallel Challenge Layer

- For each component, run critic agents in parallel:
  - Logic Critic,
  - Evidence Critic,
  - Execution Critic,
  - Optional domain specialist critic.
- Each critic returns:
  - issue,
  - severity,
  - confidence,
  - suggested fix.

### FR-4 Synthesis Layer

- Merge critiques and regenerate improved component/idea.
- Produce final outputs:
  - revised idea,
  - key changes made,
  - unresolved risks,
  - next actions.

### FR-5 Memory + Learning

- Persist run artifacts:
  - input,
  - decomposition tree,
  - critiques,
  - synthesis decisions,
  - final output,
  - user feedback.
- Enable retrieval of past patterns for future runs.

### FR-6 Modes

- Default: Deep mode.
- Optional override: Fast mode (only when explicitly requested).

### FR-7 Developer Interface

- MVP must provide at least one:
  - CLI command path, or
  - HTTP API endpoint.

### FR-8 Idempotent Run Control

- Support client-provided `run_id` to safely retry submission without duplicating runs.
- Reuse existing run when `run_id` and request payload match.
- Reject with conflict when `run_id` is reused with a different request payload.

## 9) Non-Functional Requirements

1. Reliability: system returns structured output on all valid inputs.
2. Explainability: each major revision has associated critique rationale.
3. Determinism controls: configurable temperature/seed policy.
4. Observability: log timing, token usage, and stage outcomes.
5. Configurability: depth, critic count, and thresholds are tunable.

## 10) MVP Output Contract

Each run must return JSON with this minimum schema:

```json
{
  "run_id": "string",
  "mode": "deep|fast",
  "input_idea": "string",
  "decomposition": [],
  "critic_findings": [],
  "synthesized_idea": "string",
  "changes_summary": [],
  "residual_risks": [],
  "next_experiments": [],
  "confidence_score": 0.0
}
```

## 11) High-Level System Design (MVP)

1. Orchestrator service controls pipeline and recursion.
2. LLM adapters power decomposition, challenge, and synthesis.
3. Parallel execution layer runs critics concurrently.
4. Storage layer (SQLite initially) persists runs and memory.
5. Optional API layer exposes `/analyze` and `/runs/:id`.

## 12) Success Metrics

### Product Metrics

1. Completion rate: >= 95% valid runs produce full structured output.
2. Improvement score: >= 70% of test ideas judged improved by reviewer.
3. Actionability: >= 80% runs include at least 3 concrete next steps.

### System Metrics

1. Deep mode default runtime target: 90s-5min (acceptable range).
2. Fast mode runtime target: <= 30s (when explicitly selected).
3. Pipeline stage failure rate: < 5%.

## 13) Validation Plan

1. Create benchmark set of 20 ideas across business, product, and research.
2. Run baseline single-pass analysis vs RIM MVP deep mode.
3. Blind-review outputs on rigor, novelty, actionability.
4. Track metric deltas and failure modes.

## 14) Milestones

### M1 - Foundation

- Project scaffold, config, logging, storage schema.

### M2 - Core Pipeline

- Decomposition -> parallel challenge -> synthesis chain working end-to-end.

### M3 - Memory

- Persist runs and retrieve prior insights during analysis.

### M4 - Evaluation

- Benchmark harness + quality scoring workflow.

### M5 - Hardening

- Error handling, retries, telemetry, and docs.

## 15) Risks and Mitigations

1. Latency explosion in deep recursion:
   - Mitigation: adaptive stop conditions, branch budget, caching.
2. Hallucinated critiques:
   - Mitigation: evidence requirement and confidence scoring.
3. Prompt drift across stages:
   - Mitigation: schema enforcement and stage-specific contracts.
4. Memory pollution:
   - Mitigation: score/filter retained artifacts before reuse.

## 16) Open Questions

1. None blocking for v0.2 PI migration.
2. Next decision deferred to v0.3: whether to keep multi-provider fallback enabled by default or ship PI-only default (`RIM_PI_ONLY=1`) in production environments.

## 17) MVP Completion Record

- Status: MVP scope complete
- Completion date: February 14, 2026
- Completion commit (main): `c938f09`
- Validation at completion: `35` passing tests and successful compile checks
- Latest validation snapshot (post PI-first runtime migration + PI-only mode + provider-surface updates): `163` passing tests (`pytest -q`, February 18, 2026)
- Scope basis: v0.1 milestones (M1-M5) plus FR-8 (idempotent run control)

## 18) Acceptance Checklist

- AC-1 Deep mode default in API and CLI: done
- AC-2 Recursive decomposition with stop conditions: done
- AC-3 Parallel critic execution: done
- AC-4 Multi-pass synthesis with structured output: done
- AC-5 Persistent run storage and memory reuse: done
- AC-6 Feedback loop updates memory scores and stores feedback: done
- AC-7 API endpoints for analyze, runs, logs, feedback, and health: done
- AC-8 CLI parity for analyze/run/eval workflows: done
- AC-9 Idempotent `run_id` submission (reuse + conflict detection): done
- AC-10 Benchmark baseline/compare/gate workflow: done
- AC-11 Structured error contract + partial-run behavior + retries: done
- AC-12 CI pipeline for compile + tests on push/PR: done
- AC-13 Self-iteration harness with policy and hard gates: done
- AC-14 Explicit run cancel/retry controls in API and CLI: done

## 19) Post-MVP Roadmap (v0.2)

- PI-first core refactor: replace direct Codex/Claude provider orchestration with PI as the primary execution runtime across decomposition, critique, synthesis, and benchmark baselines (done on February 18, 2026; includes `PiCLIAdapter`, PI-first default routing, strict PI-only mode via `RIM_PI_ONLY`, CLI/eval/schema/test/doc updates)
- Replace heuristic scoring with a stronger domain-weighted rubric (done on February 14, 2026)
- Finalize a canonical 20-idea benchmark pack and blind-review process (done on February 14, 2026; `rim eval blindpack`)
- Add explicit run cancel/retry controls in API and CLI (done on February 14, 2026)
- Add richer evaluation analytics (domain-level trend and regression deltas added on February 14, 2026)
- Add reusable embedding API (`rim.engine`) with composable builder functions for product integration (done on February 17, 2026)
- Add modular agent-pack registry + override surface for product-specific orchestration wiring (done on February 17, 2026)
- Add config-driven agent pack loading + env selection (`RIM_AGENT_PACKS_PATH`, `RIM_AGENT_PACK`) so API/CLI can swap orchestration packs without code edits (done on February 17, 2026)
- Add real single-call LLM baseline workflow (`rim eval baseline-llm`) for practical benchmark comparisons (done on February 17, 2026)
- Add runtime long-horizon memory quality guardrails using recent memory-fold telemetry (`RIM_ENABLE_MEMORY_QUALITY_CONTROLLER` + lookback/fold thresholds) to auto-tighten fold parameters when degradation trends increase (done on February 17, 2026)
- Add native formal constraint checks (`formal:` / `theorem:` / `constraint:`) with z3-first symbolic solving, prove/satisfiable/refute modes, assumptions (`assume=`), counterexample traces on failed proofs, and controlled AST fallback (`RIM_ADV_VERIFY_SOLVER_BACKEND`, `RIM_ADV_VERIFY_FORMAL_ALLOW_AST_FALLBACK`, `RIM_ADV_VERIFY_FORMAL_MAX_COUNT`) (done on February 17, 2026)
- Add contract-aware specialist arbitration that reuses spawn-plan role/tool contracts per flagged disagreement node and emits specialist contract metadata in arbitration outputs/logs (done on February 17, 2026)
- Add telemetry-driven specialist contract controller that adjusts spawn role boosts from recent specialist arbitration outcomes (`RIM_ENABLE_SPECIALIST_CONTRACT_CONTROLLER`, lookback/min-round/min-role thresholds) and threads adjustments into specialization spawn (`adaptive_role_boosts`) (done on February 17, 2026)
- Add offline/online policy-loop integration for specialist contract control and outcome-informed spawn role boosts: specialist policy now carries controller defaults and spawn policy training consumes specialist role outcome telemetry (`specialist_role_action_counts`, `specialist_role_avg_match_score`) (done on February 17, 2026)
- Add learned dynamic spawn token contracts (`RIM_SPAWN_DYNAMIC_ROLE_CONTRACTS`) so spawn policy training can persist token-level routing/tool contracts from successful dynamic specialists and runtime spawning can reuse them (done on February 17, 2026)
- Add offline dynamic token boost learning in spawn calibration/training (`RIM_SPAWN_DYNAMIC_TOKEN_BOOSTS`) so weighted run outcomes can directly bias dynamic specialist selection per token (done on February 17, 2026)
- Add learned default dynamic spawn contract fallback (`RIM_SPAWN_DYNAMIC_DEFAULT_CONTRACT`) so spawn policy training can persist a reusable routing/tool contract for unseen dynamic tokens at runtime (done on February 17, 2026)
- Add RL memory policy credit-assignment loop (`rim eval train-rl-memory-policy`) and RL autolearn integration so memory policies can learn fold/controller defaults from run-level memory degradation telemetry (`memory_fold_*`, `memory_quality_controller_*`) (done on February 17, 2026)
- Add HTTP-backed advanced data-reference verification path (`data: ... | url=https://...`) with explicit runtime safety controls (`RIM_ADV_VERIFY_ALLOW_HTTP_DATA`, timeout/max-bytes caps, optional host allowlist via `RIM_ADV_VERIFY_HTTP_ALLOWED_HOSTS`) for broader external evidence checks (done on February 17, 2026)
- Add bundled RL orchestration trainer (`rim eval train-rl-orchestration-policy`) and RL autolearn integration path so depth/specialist/arbitration/spawn/memory updates can be emitted as one coordinated RL bundle output (done on February 17, 2026)

## 20) SOTA Alignment Status (vs `rim_paper_4.docx`)

Status date: February 17, 2026

The MVP is complete for v0.1 scope, but full SOTA-paper parity is not yet complete.

### 20.1 Implemented

- Recursive decomposition tree with stop conditions (depth, confidence, marginal gain, runtime, branch budget).
- Parallel typed challenge layer (logic, evidence, execution, adversarial critics).
- Structured synthesis with deep-mode multi-pass refinement.
- Deep mode default across API and CLI.
- Persistent run artifacts, memory retrieval filters, and feedback-driven memory rescoring.
- Explicit run controls: cancel/retry in API (`/runs/{id}/cancel`, `/runs/{id}/retry`) and CLI (`rim run cancel`, `rim run retry`).
- Domain-weighted quality rubric and domain-level benchmark analytics (`domain_metrics`, `domain_deltas`).
- Blind-review workflow for evaluator packets (`rim eval blindpack` + anonymized packet output).
- Provider orchestration guardrails (fallbacks, retries, determinism controls, and run budgets).
- Benchmark/eval workflow with baseline, compare, and gate.
- Reusable execution runtime split from orchestration (`RimExecutionEngine` + thin `RimOrchestrator` adapter).
- Public embedding API (`rim.engine`) with `build_engine`, `build_orchestrator`, and `build_agents`.
- Modular agent-pack registry (`EngineAgentRegistry`) with validated stage overrides for external product integration.
- Config-driven pack loader (`load_agent_packs_config`) and env-based pack activation path used by API/CLI startup.
- Real single-call LLM baseline path (`rim eval baseline-llm`) for practical comparisons against normal deep-thinking model calls.
- Runtime long-horizon memory quality controller that reads recent fold telemetry and adaptively tightens fold parameters when degradation pressure rises.
- Native formal symbolic constraint checks in advanced verification (`formal:` / `theorem:` / `constraint:`) with z3-first backend, prove/satisfiable/refute modes, assumptions, bounded symbolic domains, counterexample traces, and explicit fallback controls.
- Optional HTTP-backed advanced data reference checks (`data: ... | url=https://...`) with explicit runtime safety controls (opt-in + timeout/max-bytes caps + optional host allowlist).
- Contract-aware specialist arbitration that scores spawned specialist contracts per disagreement node, injects role/tool routing context into specialist arbitration prompts, and records selected specialist contract metadata.
- Telemetry-driven specialist contract controller that learns role-level boost adjustments from recent arbitration outcomes and applies those adjustments to runtime spawn scoring.
- Eval/autolearn policy integration for specialist contract control and spawn adaptation: specialist policy can now load/save controller defaults, and spawn policy calibration/RL training consume specialist role outcome telemetry to update role boosts.
- Dynamic specialist contract learning in spawn policy loops: calibration/RL training now derive token-level spawn boosts (`RIM_SPAWN_DYNAMIC_TOKEN_BOOSTS`), routing/tool contracts for dynamic roles (`RIM_SPAWN_DYNAMIC_ROLE_CONTRACTS`), and a learned default fallback contract (`RIM_SPAWN_DYNAMIC_DEFAULT_CONTRACT`) for unseen dynamic tokens; runtime spawn planning applies role/token overrides first, then token/default contracts, then generic dynamic defaults.
- RL memory policy loop with controller-default learning: `train-rl-memory-policy` and RL autolearn now optimize fold + memory-quality-controller settings from run telemetry and can persist controller defaults in memory policy artifacts.
- Bundled RL orchestration training (`train-rl-orchestration-policy`) now packages coordinated RL policy artifacts for depth/specialist/arbitration/spawn/memory and powers RL autolearn update steps.

### 20.2 Partially Implemented

- Learning layer:
  persistent memory and feedback exist, with cycle-level memory folding into episodic/working/tool entries plus fold-version/degradation telemetry, offline memory-policy training (`rim eval train-memory-policy`), RL-style memory credit assignment training (`rim eval train-rl-memory-policy`), runtime policy loading (`RIM_MEMORY_POLICY_PATH`), autolearn-driven online policy refresh (`rim eval autolearn` with blend/RL paths), and runtime long-horizon fold guardrails from recent telemetry (`RIM_ENABLE_MEMORY_QUALITY_CONTROLLER`, `RIM_MEMORY_QUALITY_LOOKBACK_RUNS`, `RIM_MEMORY_QUALITY_MIN_FOLDS`); no fully learned memory quality meta-model for decomposition/challenge/synthesis policy updates yet.
- Orchestration depth/breadth policy:
  recursive cycle controller and heuristic DepthAllocator exist, with benchmark-driven calibration/training (`rim eval calibrate`, `rim eval train-policy`), automated online update loop (`rim eval autolearn` with `RIM_DEPTH_POLICY_PATH`), RL-style reward/advantage credit assignment (`rim eval train-rl-policy`), and bundled RL orchestration packaging across depth/specialist/arbitration/spawn/memory (`rim eval train-rl-orchestration-policy`) available; full PARL/ARPO/AEPO-grade policy optimization is still missing.
- Challenge reconciliation:
  consensus/disagreement aggregation, disagreement arbitration, confidence-triggered devil's-advocate follow-up rounds, role-diversity guardrails, specialist follow-up arbitration loops with contract-aware spawn-plan role/tool routing, benchmark telemetry capture, runtime specialist contract boost adaptation from recent arbitration telemetry, offline arbitration/specialist policy training (`rim eval train-arbitration-policy`, `rim eval train-specialist-policy` + `RIM_ARBITRATION_POLICY_PATH`, `RIM_SPECIALIST_POLICY_PATH`) including controller-default keys, automated online arbitration/specialist updates (`rim eval autolearn`), and RL-style reward/advantage arbitration + specialist credit assignment (`rim eval train-rl-policy`) are implemented; full multi-agent RL arbitration training remains missing.
- Verification layer:
  deterministic post-synthesis checks, safe executable expressions, optional timed `python_exec` checks, baseline advanced adapters (`solver:`, `simulate:`, `data:`) including optional HTTP-backed `data` URL sources with explicit runtime safety caps and optional host allowlist controls, native formal symbolic checks (`formal:`/`theorem:`/`constraint:` with z3-first backend, proof modes, assumptions, and counterexample traces), and pluggable external adapter command hooks are implemented; full theorem-prover-grade verification loops and production external integrations are still missing.
- Specialization layer:
  domain-specialist spawning and scored heuristic role-selection are implemented (with thresholded specialist budgets, rationale metadata, policy-driven tool-routing/tool-contract overrides, offline spawn-policy training via `rim eval train-spawn-policy`, RL-style spawn credit assignment via `rim eval train-rl-spawn-policy`, specialist outcome-informed role-boost updates from arbitration telemetry, dynamic token spawn-score learning via `RIM_SPAWN_DYNAMIC_TOKEN_BOOSTS`, dynamic token routing/tool contract learning via `RIM_SPAWN_DYNAMIC_ROLE_CONTRACTS`, learned default dynamic fallback contracts via `RIM_SPAWN_DYNAMIC_DEFAULT_CONTRACT`, runtime policy loading via `RIM_SPAWN_POLICY_PATH`, and autolearn-driven online spawn policy refresh), but no fully learned generative multi-role agent factory.

### 20.3 Missing / Slacking Against SOTA Paper

- Fully learned dynamic agent spawning and specialization (AgentSpawner-style policy-trained role/tool generation beyond current heuristic+RL-light role/tool policy updates).
- Full neuro-symbolic verification loops (theorem-prover-grade formal tooling, richer simulation, and production external data-backed execution).
- Fully learned long-horizon memory quality controller/meta-model and adaptive fold policy optimization across decomposition/challenge/synthesis.
- Full RL-based orchestration training (PARL/ARPO/AEPO-style multi-agent policy optimization beyond current lightweight reward/advantage credit assignment).

### 20.4 Gap-Closure Priorities

1. P0: harden recursive cycle controller + DepthAllocator thresholds with benchmark-backed calibration and automated policy updates (`rim eval calibrate`, `rim eval calibrate-loop`, `rim eval autolearn` implemented).
2. P0: evolve specialist arbitration from current heuristic+online+RL-light policy defaults (now contract-aware and telemetry-adaptive with spawn role-boost adjustments) to full RL adaptive specialist policy and learned role-contract generation.
3. P1: evolve specialization from current heuristic + offline/RL-light spawn policies + telemetry role-boost adaptation and static/dynamic token rules to fully learned dynamic role/tool generation contracts.
4. P1: extend advanced verification from local adapters to formal theorem/constraint tooling and external simulation/data integrations.
5. P2: evolve memory from current episodic/working/tool stores + online policy refresh + telemetry-driven runtime guardrails to fully learned long-horizon adaptive fold-quality optimization.
6. P3: evolve current offline heuristic policy training (`rim eval train-policy`) to learned policy optimization and credit assignment.

### 20.5 Delivery Stages for Paper Parity

1. v0.2: tune recursive cycle controller + expand reconciliation + stronger evaluation pack.
2. v0.3: specialization layer + external verification tools.
3. v0.4: tripartite memory + folding.
4. v0.5: RL orchestration training and policy rollout.

## 21) Refactor Directive: PI-First Runtime Replacement

Status date: February 18, 2026
Status: complete (v0.2 PI migration scope)

Directive: perform a total model-execution refactor so PI is the main runtime substrate for RIM.

### 21.1 Scope

1. Introduce PI as the default provider/runtime for all LLM-driven stages.
2. Keep RIM orchestration semantics (recursive flow, critics, synthesis, verification, memory) and output schema intact.
3. Preserve API and CLI interfaces unless explicitly versioned.
4. Maintain deterministic/idempotent run behavior and existing telemetry expectations.

### 21.2 Technical Outcomes Required

1. Provider layer supports PI as first-class and default execution path.
2. Stage routing defaults to PI-first ordering, with optional fallback providers.
3. Benchmark and baseline workflows accept PI as a selectable baseline provider.
4. Data models and validation allow `provider="pi"` in structured findings/logs.
5. Healthcheck and failure contracts remain stable after migration.

### 21.3 Migration Constraints

1. No regression to existing JSON output contract in section 10.
2. No regression to idempotent `run_id` behavior (FR-8).
3. Existing tests must pass or be updated only where provider-name assumptions are hard-coded.
4. Legacy Codex/Claude adapters can remain as compatibility fallbacks during migration.

### 21.4 Delivery Phases

1. Phase A: PRD + architecture alignment (this section) and provider contract updates. (done on February 18, 2026)
2. Phase B: PI adapter integration + router defaults + CLI/eval provider surfaces. (done on February 18, 2026)
3. Phase C: test suite updates for provider literals/order assumptions. (done on February 18, 2026)
4. Phase D: hardening, docs, and rollout guidance for PI subscription/API-key setups. (done on February 18, 2026)

### 21.5 Completion Evidence

1. PI provider adapter added and wired as first-class runtime backend.
2. Provider routing defaults switched to PI-first ordering with configurable `RIM_PROVIDER_ORDER`.
3. Strict PI-only runtime mode implemented (`RIM_PI_ONLY=1`) to disable Codex/Claude fallback.
4. Eval/CLI surfaces updated to accept PI for baseline workflows.
5. Schema and provider assertions updated to support `provider="pi"`.
6. Validation completed: full test suite passing (`163` tests, February 18, 2026).

## 22) SOTA Watchlist Update (from VoltAgent 2026 list)

Status date: February 18, 2026

This shortlist is prioritized for the next SOTA paper revision and v0.3-v0.5 execution planning.

1. [ROMA: Recursive Open Meta-Agent Framework for Long-Horizon Multi-Agent Systems](https://arxiv.org/abs/2602.01848v1)
   - RIM relevance: strengthens recursive long-horizon decomposition and subtask-tree execution strategy.
2. [ORCH: many analyses, one merge - a deterministic multi-agent orchestrator](https://arxiv.org/abs/2602.01797v1)
   - RIM relevance: aligns with deterministic multi-provider merge behavior and arbitration simplification.
3. [Learning Latency-Aware Orchestration for Parallel Multi-Agent Systems](https://arxiv.org/abs/2601.10560v1)
   - RIM relevance: informs runtime critical-path optimization for deep mode latency control.
4. [Learning to Recommend Multi-Agent Subgraphs from Calling Trees](https://arxiv.org/abs/2601.22209v1)
   - RIM relevance: directly maps to learned specialist spawning/routing from historical run telemetry.
5. [BudgetMem: Learning Query-Aware Budget-Tier Routing for Runtime Agent Memory](https://arxiv.org/abs/2602.06025v1)
   - RIM relevance: complements memory-quality controller with budget-aware memory retrieval tiers.
6. [Learning to Share: Selective Memory for Efficient Parallel Agentic Systems](https://arxiv.org/abs/2602.05965v1)
   - RIM relevance: improves parallel critic memory exchange without token bloat.
7. [ProcMEM: Learning Reusable Procedural Memory from Experience via Non-Parametric PPO for LLM Agents](https://arxiv.org/abs/2602.01869v1)
   - RIM relevance: supports reusable procedural memory artifacts for recurring reasoning patterns.
8. [Replayable Financial Agents: A Determinism-Faithfulness Assurance Harness for Tool-Using LLM Agents](https://arxiv.org/abs/2601.15322v1)
   - RIM relevance: adds a direct blueprint for deterministic trajectory replay and faithfulness checks.
9. [AEMA: Verifiable Evaluation Framework for Trustworthy and Controlled Agentic LLM Systems](https://arxiv.org/abs/2601.11903v1)
   - RIM relevance: strengthens process-aware auditable eval pipeline design.
10. [AgenTRIM: Tool Risk Mitigation for Agentic AI](https://arxiv.org/abs/2601.12449v1)
   - RIM relevance: maps to runtime least-privilege tool contracts and advanced verification hardening.

Execution note:
- Incorporate these references into the next SOTA manuscript revision (`rim_paper_5` target) and use them as priors for v0.3-v0.5 implementation scope updates.
