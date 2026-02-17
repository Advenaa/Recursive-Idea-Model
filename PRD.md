# Product Requirements Document (PRD)

## Product

Recursive Idea Model (RIM) MVP

## Version

v0.1 (updated February 17, 2026)

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

1. First domain focus for evaluation (general vs specific vertical)?
2. Preferred LLM/provider stack for MVP implementation?
3. Quality scoring rubric weights (rigor vs novelty vs practicality)?
4. Minimum acceptable deep mode runtime for your workflow?

## 17) MVP Completion Record

- Status: MVP scope complete
- Completion date: February 14, 2026
- Completion commit (main): `c938f09`
- Validation at completion: `35` passing tests and successful compile checks
- Latest validation snapshot (post-v0.2 + RL-light depth/arbitration/spawn + engine modularization increments): `150` passing tests (`pytest -q`, February 17, 2026)
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
- Contract-aware specialist arbitration that scores spawned specialist contracts per disagreement node, injects role/tool routing context into specialist arbitration prompts, and records selected specialist contract metadata.
- Telemetry-driven specialist contract controller that learns role-level boost adjustments from recent arbitration outcomes and applies those adjustments to runtime spawn scoring.

### 20.2 Partially Implemented

- Learning layer:
  persistent memory and feedback exist, with cycle-level memory folding into episodic/working/tool entries plus fold-version/degradation telemetry, offline memory-policy training (`rim eval train-memory-policy`), runtime policy loading (`RIM_MEMORY_POLICY_PATH`), autolearn-driven online policy refresh (`rim eval autolearn`), and runtime long-horizon fold guardrails from recent telemetry (`RIM_ENABLE_MEMORY_QUALITY_CONTROLLER`, `RIM_MEMORY_QUALITY_LOOKBACK_RUNS`, `RIM_MEMORY_QUALITY_MIN_FOLDS`); no fully learned memory quality meta-model for decomposition/challenge/synthesis policy updates yet.
- Orchestration depth/breadth policy:
  recursive cycle controller and heuristic DepthAllocator exist, with benchmark-driven calibration/training (`rim eval calibrate`, `rim eval train-policy`), automated online update loop (`rim eval autolearn` with `RIM_DEPTH_POLICY_PATH`), and RL-style reward/advantage credit assignment (`rim eval train-rl-policy`) available; full PARL/ARPO/AEPO-grade policy optimization is still missing.
- Challenge reconciliation:
  consensus/disagreement aggregation, disagreement arbitration, confidence-triggered devil's-advocate follow-up rounds, role-diversity guardrails, specialist follow-up arbitration loops with contract-aware spawn-plan role/tool routing, benchmark telemetry capture, runtime specialist contract boost adaptation from recent arbitration telemetry, offline arbitration/specialist policy training (`rim eval train-arbitration-policy`, `rim eval train-specialist-policy` + `RIM_ARBITRATION_POLICY_PATH`, `RIM_SPECIALIST_POLICY_PATH`), automated online arbitration/specialist updates (`rim eval autolearn`), and RL-style reward/advantage arbitration + specialist credit assignment (`rim eval train-rl-policy`) are implemented; full multi-agent RL arbitration training remains missing.
- Verification layer:
  deterministic post-synthesis checks, safe executable expressions, optional timed `python_exec` checks, baseline advanced adapters (`solver:`, `simulate:`, `data:`), native formal symbolic checks (`formal:`/`theorem:`/`constraint:` with z3-first backend, proof modes, assumptions, and counterexample traces), and pluggable external adapter command hooks are implemented; full theorem-prover-grade verification loops and production external integrations are still missing.
- Specialization layer:
  domain-specialist spawning and scored heuristic role-selection are implemented (with thresholded specialist budgets, rationale metadata, policy-driven tool-routing/tool-contract overrides, offline spawn-policy training via `rim eval train-spawn-policy`, RL-style spawn credit assignment via `rim eval train-rl-spawn-policy`, runtime policy loading via `RIM_SPAWN_POLICY_PATH`, and autolearn-driven online spawn policy refresh), but no fully learned generative multi-role agent factory.

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
