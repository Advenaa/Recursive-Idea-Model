# Self-Iteration Policy

## Purpose

Use RIM to improve the RIM codebase with controlled, auditable loops.

## Safety Rules

- Scope each iteration to one objective and a small file/module set.
- Keep a single iteration branch; do not batch unrelated refactors.
- Require deterministic run tracking with explicit `run_id`.
- Require all gates to pass before merge:
  - compile,
  - tests,
  - eval regression gate.
- Reject changes that only optimize for passing tests without preserving behavior.

## Loop

1. Prepare
   - command:
     - `scripts/self_iteration.sh prepare --objective "<objective>"`
   - outputs:
     - `proposal.json` from `rim analyze`
     - compile/test baseline log
     - baseline eval report (`before_target.json`)

2. Implement
   - apply code changes from proposal with human review.
   - keep scope aligned with the objective.

3. Verify
   - command:
     - `scripts/self_iteration.sh verify --iteration-dir <dir>`
   - gates:
     - compile + tests must pass
     - eval gate must pass against baseline report

4. Review and Merge
   - summarize behavior changes and risks.
   - merge only after gate pass and reviewer approval.

## Recommended Settings

- `RIM_DETERMINISM_MODE=strict`
- `RIM_DETERMINISM_SEED=42`
- `RIM_PROVIDER_MAX_RETRIES=2`
- `RIM_PROVIDER_RETRY_BASE_MS=250`

## Escalation Rules

- If eval gate fails, do not merge.
- If structured error indicates `retryable=false` on critical stages repeatedly, stop and investigate provider adapter behavior before continuing iterations.
- If changes exceed agreed scope, split into a new iteration objective.
