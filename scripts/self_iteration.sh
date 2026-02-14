#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RIM_BIN="${RIM_BIN:-$ROOT_DIR/.venv/bin/rim}"
PYTEST_BIN="${PYTEST_BIN:-$ROOT_DIR/.venv/bin/pytest}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DEFAULT_OUT_DIR="$ROOT_DIR/artifacts/self-iterations"
DEFAULT_MODE="deep"
DEFAULT_LIMIT="3"
DEFAULT_MIN_QUALITY_DELTA="0.0"
DEFAULT_MAX_RUNTIME_DELTA_SEC="15"
DEFAULT_MIN_SHARED_RUNS="1"

usage() {
  cat <<'EOF'
Usage:
  scripts/self_iteration.sh prepare --objective "..." [options]
  scripts/self_iteration.sh verify --iteration-dir <dir> [options]

Commands:
  prepare   Generate a proposal and baseline artifacts for one iteration.
  verify    Run quality gates for the iteration after code edits.

Prepare options:
  --objective <text>            Required objective to analyze.
  --run-id <id>                 Optional stable run_id for proposal generation.
  --out-dir <dir>               Output root (default: artifacts/self-iterations).
  --mode <deep|fast>            Eval mode for baseline report (default: deep).
  --limit <n>                   Eval dataset limit (default: 3).
  --skip-eval                   Skip eval report generation in prepare.
  --allow-dirty                 Allow prepare to run with uncommitted changes.

Verify options:
  --iteration-dir <dir>         Required iteration directory from prepare phase.
  --mode <deep|fast>            Eval mode for after-change report (default: deep).
  --limit <n>                   Eval dataset limit (default: 3).
  --min-quality-delta <float>   Gate threshold (default: 0.0).
  --max-runtime-delta-sec <f>   Gate threshold (default: 15).
  --min-shared-runs <n>         Gate threshold (default: 1).
  --skip-eval                   Skip eval gate during verify.

Examples:
  scripts/self_iteration.sh prepare --objective "Improve run queue reliability"
  scripts/self_iteration.sh verify --iteration-dir artifacts/self-iterations/20260214_120000_queue-reliability
EOF
}

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "missing required command: $cmd" >&2
    exit 1
  fi
}

slugify() {
  local input="$1"
  echo "$input" \
    | tr '[:upper:]' '[:lower:]' \
    | sed 's/[^a-z0-9]/-/g' \
    | sed 's/-\{2,\}/-/g' \
    | sed 's/^-//' \
    | sed 's/-$//' \
    | cut -c1-48
}

git_clean() {
  if [[ -n "$(git -C "$ROOT_DIR" status --porcelain)" ]]; then
    return 1
  fi
  return 0
}

run_compile_and_tests() {
  echo "[gate] compile"
  "$PYTHON_BIN" -m compileall "$ROOT_DIR/rim" "$ROOT_DIR/tests"
  echo "[gate] tests"
  "$PYTEST_BIN" -q
}

prepare_cmd() {
  local objective=""
  local run_id=""
  local out_dir="$DEFAULT_OUT_DIR"
  local mode="$DEFAULT_MODE"
  local limit="$DEFAULT_LIMIT"
  local skip_eval="false"
  local allow_dirty="false"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --objective)
        objective="${2:-}"
        shift 2
        ;;
      --run-id)
        run_id="${2:-}"
        shift 2
        ;;
      --out-dir)
        out_dir="${2:-}"
        shift 2
        ;;
      --mode)
        mode="${2:-}"
        shift 2
        ;;
      --limit)
        limit="${2:-}"
        shift 2
        ;;
      --skip-eval)
        skip_eval="true"
        shift
        ;;
      --allow-dirty)
        allow_dirty="true"
        shift
        ;;
      *)
        echo "unknown option for prepare: $1" >&2
        usage
        exit 1
        ;;
    esac
  done

  if [[ -z "$objective" ]]; then
    echo "--objective is required for prepare" >&2
    usage
    exit 1
  fi

  if [[ "$allow_dirty" != "true" ]] && ! git_clean; then
    echo "prepare requires a clean git tree (use --allow-dirty to override)" >&2
    exit 1
  fi

  local timestamp
  timestamp="$(date -u +%Y%m%d_%H%M%S)"
  local slug
  slug="$(slugify "$objective")"
  if [[ -z "$slug" ]]; then
    slug="iteration"
  fi
  local iteration_dir="$out_dir/${timestamp}_${slug}"
  mkdir -p "$iteration_dir"

  local base_sha
  base_sha="$(git -C "$ROOT_DIR" rev-parse HEAD)"
  local proposal_run_id
  proposal_run_id="${run_id:-self-iter-${timestamp}}"

  cat >"$iteration_dir/context.txt" <<EOF
objective: $objective
created_at_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)
base_sha: $base_sha
proposal_run_id: $proposal_run_id
mode: $mode
limit: $limit
skip_eval: $skip_eval
EOF

  echo "[prepare] generating proposal"
  "$RIM_BIN" analyze \
    --idea "$objective" \
    --mode deep \
    --run-id "$proposal_run_id" \
    --json \
    --save "$iteration_dir/proposal.json" \
    >"$iteration_dir/proposal.out.json"

  run_compile_and_tests >"$iteration_dir/precheck.log" 2>&1

  if [[ "$skip_eval" != "true" ]]; then
    echo "[prepare] baseline eval report"
    "$RIM_BIN" eval run \
      --mode "$mode" \
      --limit "$limit" \
      --save "$iteration_dir/before_target.json" \
      >"$iteration_dir/before_eval.out.json"
  fi

  echo "iteration_dir=$iteration_dir"
  echo "next: apply code changes, then run verify phase."
}

verify_cmd() {
  local iteration_dir=""
  local mode="$DEFAULT_MODE"
  local limit="$DEFAULT_LIMIT"
  local min_quality_delta="$DEFAULT_MIN_QUALITY_DELTA"
  local max_runtime_delta_sec="$DEFAULT_MAX_RUNTIME_DELTA_SEC"
  local min_shared_runs="$DEFAULT_MIN_SHARED_RUNS"
  local skip_eval="false"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --iteration-dir)
        iteration_dir="${2:-}"
        shift 2
        ;;
      --mode)
        mode="${2:-}"
        shift 2
        ;;
      --limit)
        limit="${2:-}"
        shift 2
        ;;
      --min-quality-delta)
        min_quality_delta="${2:-}"
        shift 2
        ;;
      --max-runtime-delta-sec)
        max_runtime_delta_sec="${2:-}"
        shift 2
        ;;
      --min-shared-runs)
        min_shared_runs="${2:-}"
        shift 2
        ;;
      --skip-eval)
        skip_eval="true"
        shift
        ;;
      *)
        echo "unknown option for verify: $1" >&2
        usage
        exit 1
        ;;
    esac
  done

  if [[ -z "$iteration_dir" ]]; then
    echo "--iteration-dir is required for verify" >&2
    usage
    exit 1
  fi
  if [[ ! -d "$iteration_dir" ]]; then
    echo "iteration directory not found: $iteration_dir" >&2
    exit 1
  fi

  run_compile_and_tests >"$iteration_dir/postcheck.log" 2>&1

  if [[ "$skip_eval" == "true" ]]; then
    echo "[verify] skipped eval gate (compile/tests only)"
    return 0
  fi

  local base_report="$iteration_dir/before_target.json"
  if [[ ! -f "$base_report" ]]; then
    echo "missing baseline report: $base_report" >&2
    echo "rerun prepare without --skip-eval or provide baseline manually." >&2
    exit 1
  fi

  echo "[verify] after-change eval report"
  "$RIM_BIN" eval run \
    --mode "$mode" \
    --limit "$limit" \
    --save "$iteration_dir/after_target.json" \
    >"$iteration_dir/after_eval.out.json"

  echo "[verify] regression gate"
  set +e
  "$RIM_BIN" eval gate \
    --base "$base_report" \
    --target "$iteration_dir/after_target.json" \
    --min-quality-delta "$min_quality_delta" \
    --max-runtime-delta-sec "$max_runtime_delta_sec" \
    --min-shared-runs "$min_shared_runs" \
    >"$iteration_dir/gate.out.json"
  local gate_exit=$?
  set -e

  cat "$iteration_dir/gate.out.json"
  if [[ "$gate_exit" -ne 0 ]]; then
    echo "gate failed for iteration: $iteration_dir" >&2
    exit "$gate_exit"
  fi
  echo "[verify] gate passed"
}

main() {
  require_cmd git
  require_cmd "$PYTHON_BIN"
  require_cmd "$RIM_BIN"
  require_cmd "$PYTEST_BIN"

  if [[ $# -lt 1 ]]; then
    usage
    exit 1
  fi

  local command="$1"
  shift
  case "$command" in
    prepare)
      prepare_cmd "$@"
      ;;
    verify)
      verify_cmd "$@"
      ;;
    -h|--help|help)
      usage
      ;;
    *)
      echo "unknown command: $command" >&2
      usage
      exit 1
      ;;
  esac
}

main "$@"
