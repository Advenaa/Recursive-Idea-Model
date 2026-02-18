# The Recursive Idea Model (RIM) v5

## PI-First Recursive Multi-Agent Orchestration with Deterministic Controls, Verifiable Evaluation, and Memory-Aware Adaptation

Version: v5 (February 18, 2026)  
Codebase: `Advenaa/Recursive-Idea-Model` (`main`)  
Validation snapshot: `163` passing tests (`pytest -q`, February 18, 2026)

## Abstract

RIM v5 is a production-oriented recursive multi-agent reasoning system that transforms raw ideas into stronger, stress-tested outputs through decomposition, adversarial challenge, synthesis, verification, and memory-informed refinement.  

This revision moves the runtime substrate to a PI-first provider architecture while preserving the existing run contract and orchestration semantics. The system now supports: (1) PI-first stage routing, (2) strict PI-only execution mode, (3) deterministic retry/JSON-repair controls, (4) explicit run budgets and telemetry, and (5) benchmark/evaluation workflows for single-pass and multi-stage comparisons.

RIM v5 remains focused on developer-facing reliability: idempotent run control, structured failure contracts, replayable stage logs, and policy-driven adaptation for depth, arbitration, specialization, and memory quality.

## 1. Motivation

Single-pass LLM outputs are often coherent but shallow. Real-world idea development requires:

1. recursive decomposition of claims and assumptions,
2. adversarial challenge from multiple perspectives,
3. synthesis under explicit constraints,
4. verifiable outputs with traceable rationale,
5. feedback-aware memory that improves over repeated runs.

RIM addresses these with a staged recursive pipeline and explicit operational controls over determinism, budget, and failure handling.

## 2. Contributions in v5

1. PI-first execution architecture
   - Added `PiCLIAdapter` and routed runtime defaults to `pi,codex,claude`.
   - Added strict PI-only mode (`RIM_PI_ONLY=1`) to disable fallback providers.
2. Stable provider contract under migration
   - Preserved run/result schemas and stage semantics during provider substrate transition.
   - Extended provider literals and baseline surfaces to include `pi`.
3. Deterministic and budgeted orchestration controls
   - Determinism hints, staged retries, JSON repair retries, and budget gates for calls/tokens/latency/cost.
4. Operationally grounded eval stack
   - Supports deterministic baseline, single-call LLM baselines, duel comparisons, and regression gates.
5. Evidence-backed implementation maturity
   - Full test suite passing at 163 tests after PI-first migration.

## 3. System Architecture

RIM v5 implements a recursive cycle with bounded branching and optional multi-cycle depth control.

1. Intake
   - API/CLI accepts idea, mode, optional domain/constraints/outcome.
2. Decomposition
   - Expands root into claims/assumptions/dependencies with stop conditions.
3. Parallel critics
   - Logic, evidence, execution, adversarial (and optional domain specialist) critiques.
4. Reconciliation + arbitration
   - Aggregates findings, resolves disagreements, supports specialist follow-up arbitration.
5. Synthesis
   - Produces revised idea, deltas, residual risks, and next experiments.
6. Verification
   - Structured checks, optional executable checks, advanced adapters (`solver:`, `simulate:`, `data:`, formal constraints).
7. Memory folding + persistence
   - Stores run artifacts and feedback-linked memory entries with quality controls.

Core implementation modules:

- Orchestration/runtime: `rim/core/orchestrator.py`, `rim/core/engine_runtime.py`
- Agents: `rim/agents/*.py`
- Provider routing: `rim/providers/router.py`
- PI adapter: `rim/providers/pi_cli.py`
- API: `rim/api/app.py`
- Persistence: `rim/storage/repo.py`, `rim/storage/db.py`
- Evaluation: `rim/eval/runner.py`

## 4. PI-First Provider Layer

RIM v5 introduces PI as first-class provider runtime.

1. Default provider order
   - `RIM_PROVIDER_ORDER=pi,codex,claude`
2. Strict PI-only mode
   - `RIM_PI_ONLY=1` forces PI-only stage execution
3. PI adapter controls
   - `RIM_PI_CMD`, `RIM_PI_ARGS`, `RIM_PI_PROVIDER`, `RIM_PI_MODEL`, `RIM_PI_THINKING`
4. Fallback/repair behavior
   - transient retry with backoff
   - JSON repair retries when parse/schema shape fails
   - budget checks before and after each provider call

This design allows runtime migration to PI without changing higher-level orchestration logic.

## 5. Determinism, Reliability, and Failure Semantics

RIM v5 keeps explicit reliability contracts:

1. Idempotent run control
   - Client `run_id` reuse with payload match and conflict detection.
2. Structured errors
   - `{ stage, provider, message, retryable }`
3. Run statuses
   - `queued`, `running`, `completed`, `failed`, `partial`, `canceled`
4. Observability
   - Stage logs include status/provider/latency/meta
5. Determinism controls
   - `RIM_DETERMINISM_MODE`, `RIM_DETERMINISM_SEED`

## 6. Learning and Policy Adaptation

RIM v5 includes online/offline adaptation paths:

1. Depth policy calibration/training
2. Arbitration/specialist policy training
3. Spawn policy training with role/tool contracts
4. Memory policy training, including RL-style credit assignment
5. Autolearn loop for policy refresh from benchmark reports

Memory quality controls include runtime guardrails and fold telemetry-based adjustment.

## 7. Evaluation Framework

RIM supports three benchmark regimes:

1. Deterministic single-pass baseline (`rim eval baseline`)
2. Single-call LLM baseline (`rim eval baseline-llm --provider pi|claude|codex`)
3. Full orchestrated benchmark (`rim eval run`)

Comparison/gating:

1. `rim eval compare`
2. `rim eval gate`
3. `rim eval duel` for baseline+target+gate in one flow

This enables tracking quality/runtime deltas and regression enforcement.

## 8. Current SOTA Position (v5)

RIM v5 is strong in:

1. recursive orchestration with explicit budgets and failure controls,
2. deterministic control surfaces and idempotent execution contracts,
3. multi-stage reconciliation + specialist arbitration loops,
4. practical benchmark and policy calibration workflows,
5. PI-first runtime migration with compatibility fallback path.

Gaps still open for next major revision:

1. fully learned multi-agent routing/topology generation,
2. theorem-prover-grade verification loops beyond current formal checks,
3. richer long-horizon learned memory controllers across all stages,
4. stronger security-hardening defaults for tool access and prompt injection resistance.

## 9. New 2026 SOTA Paper Priorities for v6

The following papers are now prioritized for direct integration into the next paper/release cycle:

1. ROMA (recursive long-horizon multi-agent decomposition): https://arxiv.org/abs/2602.01848v1
2. ORCH (deterministic merge orchestration): https://arxiv.org/abs/2602.01797v1
3. Learning Latency-Aware Orchestration: https://arxiv.org/abs/2601.10560v1
4. Learning to Recommend Multi-Agent Subgraphs from Calling Trees: https://arxiv.org/abs/2601.22209v1
5. BudgetMem: https://arxiv.org/abs/2602.06025v1
6. Learning to Share (selective memory sharing): https://arxiv.org/abs/2602.05965v1
7. ProcMEM: https://arxiv.org/abs/2602.01869v1
8. Replayable Financial Agents (determinism-faithfulness harness): https://arxiv.org/abs/2601.15322v1
9. AEMA (verifiable process-aware evaluation): https://arxiv.org/abs/2601.11903v1
10. AgenTRIM (tool risk mitigation): https://arxiv.org/abs/2601.12449v1

Planned integration themes:

1. learned routing from calling-tree telemetry,
2. latency-aware critical-path optimization in deep mode,
3. budget-tier memory controller upgrades,
4. replayable trajectory faithfulness auditing,
5. stricter runtime tool-risk policies.

## 10. Conclusion

RIM v5 establishes a stable PI-first foundation while preserving recursive analysis quality, structured contracts, and evaluation discipline. The system is now positioned for the next SOTA jump: learned orchestration policies, stronger memory controllers, verifiable evaluation rigor, and hardened tool-security boundaries.

This release is best viewed as an execution-layer consolidation milestone that prepares the architecture for higher-order learning and safety upgrades in v6.
