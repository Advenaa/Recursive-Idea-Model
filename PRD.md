# Product Requirements Document (PRD)

## Product

Recursive Idea Model (RIM) MVP

## Version

v0.1 (February 14, 2026)

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

## 17) Implementation Status (February 14, 2026)

### Milestone Status

- M1 Foundation: completed
- M2 Core Pipeline: completed
- M3 Memory: completed
- M4 Evaluation: completed for benchmark, baseline comparator, report compare, and regression gate
- M5 Hardening: completed for retries, structured error contract, partial-run behavior, telemetry logs, and CI

### Delivered Capabilities

- Deep mode default with recursive decomposition + branch/runtime controls
- Parallel critic stage and multi-pass synthesis
- Persistent run artifacts, memory reuse, and feedback loop
- API endpoints: `/analyze`, `/runs`, `/runs/{run_id}`, `/runs/{run_id}/logs`, `/runs/{run_id}/feedback`, `/health`
- Idempotent API submission with `run_id`
- CLI coverage for analyze, run inspection/list/logs/feedback, and eval workflows
- Evaluation workflows: `eval run`, `eval baseline`, `eval compare`, `eval gate`, `eval duel`
- GitHub Actions CI for compile + tests on push/PR

### Remaining Post-MVP Enhancements

- Replace heuristic scoring with a stronger domain-weighted rubric
- Expand benchmark set to a stable 20-idea canonical pack with blind-review protocol
- Add explicit run cancel/retry controls
