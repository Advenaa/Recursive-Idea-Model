from __future__ import annotations

import ast
import json
import os
import shlex
import subprocess
import random
import re
from pathlib import Path
from typing import Any

from rim.core.schemas import CriticFinding

_SOLVER_PREFIXES = ("solver:", "solve:")
_FORMAL_SOLVER_PREFIXES = ("formal:", "theorem:", "constraint:")
_SIM_PREFIXES = ("simulate:", "simulation:")
_DATA_PREFIXES = ("data:", "dataset:")
_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")

_ALLOWED_AST_NODES = (
    ast.Expression,
    ast.BoolOp,
    ast.BinOp,
    ast.UnaryOp,
    ast.Compare,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.And,
    ast.Or,
    ast.Not,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.Pow,
    ast.FloorDiv,
    ast.USub,
    ast.UAdd,
    ast.Eq,
    ast.NotEq,
    ast.Gt,
    ast.GtE,
    ast.Lt,
    ast.LtE,
    ast.In,
    ast.NotIn,
    ast.Is,
    ast.IsNot,
)


def _extract_advanced_checks(constraints: list[str] | None, max_checks: int) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    for item in list(constraints or []):
        text = str(item).strip()
        if not text:
            continue
        lowered = text.lower()
        matched: dict[str, str] | None = None
        for prefix in _FORMAL_SOLVER_PREFIXES:
            if lowered.startswith(prefix):
                matched = {"kind": "formal_solver", "payload": text[len(prefix) :].strip()}
                break
        for prefix in _SOLVER_PREFIXES:
            if lowered.startswith(prefix):
                matched = {"kind": "solver", "payload": text[len(prefix) :].strip()}
                break
        if matched is None:
            for prefix in _SIM_PREFIXES:
                if lowered.startswith(prefix):
                    matched = {"kind": "simulation", "payload": text[len(prefix) :].strip()}
                    break
        if matched is None:
            for prefix in _DATA_PREFIXES:
                if lowered.startswith(prefix):
                    matched = {"kind": "data_reference", "payload": text[len(prefix) :].strip()}
                    break
        if matched is None:
            continue
        if matched["payload"]:
            checks.append(matched)
        if len(checks) >= max(1, int(max_checks)):
            break
    return checks


def _validate_ast(node: ast.AST, allowed_names: set[str]) -> None:
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            raise ValueError("Function calls are not allowed in advanced checks.")
        if isinstance(child, ast.Attribute):
            raise ValueError("Attribute access is not allowed in advanced checks.")
        if isinstance(child, ast.Subscript):
            raise ValueError("Subscript access is not allowed in advanced checks.")
        if not isinstance(child, _ALLOWED_AST_NODES):
            raise ValueError(f"Unsupported syntax in advanced check: {type(child).__name__}")
        if isinstance(child, ast.Name) and child.id not in allowed_names:
            raise ValueError(f"Unknown variable '{child.id}' in advanced check.")


def _safe_eval_expression(expression: str, context: dict[str, Any]) -> bool:
    parsed = ast.parse(expression, mode="eval")
    _validate_ast(parsed, set(context.keys()))
    compiled = compile(parsed, "<rim-advanced-check>", "eval")
    value = eval(compiled, {"__builtins__": {}}, dict(context))  # noqa: S307
    return bool(value)


def _z3_from_ast(node: ast.AST, context: dict[str, Any], z3: Any) -> Any:  # noqa: ANN401
    if isinstance(node, ast.Expression):
        return _z3_from_ast(node.body, context, z3)
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, bool):
            return z3.BoolVal(value)
        if isinstance(value, (int, float)):
            return z3.RealVal(float(value))
        raise ValueError("unsupported constant type for z3 backend")
    if isinstance(node, ast.Name):
        if node.id not in context:
            raise ValueError(f"Unknown variable '{node.id}' in advanced check.")
        value = context[node.id]
        if isinstance(value, bool):
            return z3.BoolVal(value)
        if isinstance(value, (int, float)):
            return z3.RealVal(float(value))
        raise ValueError(f"Unsupported variable type for '{node.id}' in z3 backend")
    if isinstance(node, ast.UnaryOp):
        value = _z3_from_ast(node.operand, context, z3)
        if isinstance(node.op, ast.Not):
            return z3.Not(value)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return value
        raise ValueError("unsupported unary operator for z3 backend")
    if isinstance(node, ast.BinOp):
        left = _z3_from_ast(node.left, context, z3)
        right = _z3_from_ast(node.right, context, z3)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Mod):
            return z3.Mod(left, right)
        if isinstance(node.op, ast.Pow):
            return left**right
        if isinstance(node.op, ast.FloorDiv):
            return left / right
        raise ValueError("unsupported binary operator for z3 backend")
    if isinstance(node, ast.BoolOp):
        values = [_z3_from_ast(item, context, z3) for item in node.values]
        if isinstance(node.op, ast.And):
            return z3.And(*values)
        if isinstance(node.op, ast.Or):
            return z3.Or(*values)
        raise ValueError("unsupported boolean operator for z3 backend")
    if isinstance(node, ast.Compare):
        left = _z3_from_ast(node.left, context, z3)
        comparisons: list[Any] = []
        cursor = left
        for op, comparator in zip(node.ops, node.comparators, strict=False):
            right = _z3_from_ast(comparator, context, z3)
            if isinstance(op, ast.Eq):
                comparisons.append(cursor == right)
            elif isinstance(op, ast.NotEq):
                comparisons.append(cursor != right)
            elif isinstance(op, ast.Gt):
                comparisons.append(cursor > right)
            elif isinstance(op, ast.GtE):
                comparisons.append(cursor >= right)
            elif isinstance(op, ast.Lt):
                comparisons.append(cursor < right)
            elif isinstance(op, ast.LtE):
                comparisons.append(cursor <= right)
            else:
                raise ValueError("unsupported comparison operator for z3 backend")
            cursor = right
        return z3.And(*comparisons) if comparisons else z3.BoolVal(True)
    raise ValueError(f"unsupported syntax for z3 backend: {type(node).__name__}")


def _safe_eval_z3(expression: str, context: dict[str, Any]) -> bool:
    try:
        import z3  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"z3 backend unavailable: {exc}") from exc
    parsed = ast.parse(expression, mode="eval")
    _validate_ast(parsed, set(context.keys()))
    expr = _z3_from_ast(parsed, context, z3)
    solver = z3.Solver()
    solver.add(expr)
    return solver.check() == z3.sat


def _solver_backend() -> str:
    backend = str(os.getenv("RIM_ADV_VERIFY_SOLVER_BACKEND", "ast")).strip().lower()
    if backend in {"ast", "z3"}:
        return backend
    return "ast"


def _parse_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _verification_context(
    *,
    synthesis: dict[str, object],
    findings: list[CriticFinding],
) -> dict[str, Any]:
    high_count = len([item for item in findings if item.severity == "high"])
    critical_count = len([item for item in findings if item.severity == "critical"])
    return {
        "confidence_score": float(synthesis.get("confidence_score", 0.0) or 0.0),
        "change_count": len(list(synthesis.get("changes_summary") or [])),
        "risk_count": len(list(synthesis.get("residual_risks") or [])),
        "experiment_count": len(list(synthesis.get("next_experiments") or [])),
        "finding_count": len(findings),
        "high_finding_count": high_count,
        "critical_finding_count": critical_count,
    }


def _parse_option_value(raw: str) -> tuple[str, str] | None:
    if "=" not in raw:
        return None
    key, value = raw.split("=", 1)
    key_text = str(key).strip().lower()
    value_text = str(value).strip()
    if not key_text or not value_text:
        return None
    return key_text, value_text


def _split_payload(payload: str) -> tuple[str, dict[str, str]]:
    segments = [str(item).strip() for item in str(payload).split("|") if str(item).strip()]
    if not segments:
        return "", {}
    expression = segments[0]
    options: dict[str, str] = {}
    for item in segments[1:]:
        parsed = _parse_option_value(item)
        if parsed is None:
            continue
        key, value = parsed
        options[key] = value
    return expression, options


def _synthesis_text(synthesis: dict[str, object]) -> str:
    parts: list[str] = [str(synthesis.get("synthesized_idea") or "").strip()]
    parts.extend(str(item).strip() for item in list(synthesis.get("changes_summary") or []) if str(item).strip())
    parts.extend(
        str(item).strip()
        for item in list(synthesis.get("next_experiments") or [])
        if str(item).strip()
    )
    return " ".join(item for item in parts if item)


def _tokenize(text: str) -> set[str]:
    return {item.group(0).lower() for item in _TOKEN_RE.finditer(str(text))}


def _run_solver_check(
    expression: str,
    context: dict[str, Any],
    *,
    backend: str = "ast",
    allow_ast_fallback: bool = True,
    check_type: str = "solver",
) -> dict[str, object]:
    backend_name = str(backend).strip().lower()
    if backend_name not in {"ast", "z3"}:
        backend_name = "ast"
    try:
        if backend_name == "z3":
            passed = _safe_eval_z3(expression, context)
            return {
                "check_type": check_type,
                "expression": expression,
                "passed": passed,
                "result": {"passed": passed, "backend": "z3"},
                "error": None,
            }
        passed = _safe_eval_expression(expression, context)
        return {
            "check_type": check_type,
            "expression": expression,
            "passed": passed,
            "result": {"passed": passed, "backend": "ast"},
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        if backend_name == "z3" and allow_ast_fallback:
            try:
                passed = _safe_eval_expression(expression, context)
                return {
                    "check_type": check_type,
                    "expression": expression,
                    "passed": passed,
                    "result": {"passed": passed, "backend": "ast_fallback"},
                    "error": None,
                }
            except Exception:  # noqa: BLE001
                pass
        return {
            "check_type": check_type,
            "expression": expression,
            "passed": False,
            "result": None,
            "error": str(exc),
        }


def _normalize_external_command(command: str | None) -> list[str] | None:
    raw = str(command or "").strip()
    if not raw:
        return None
    try:
        parsed = shlex.split(raw)
    except ValueError:
        return None
    return parsed if parsed else None


def _run_external_adapter(
    *,
    check_type: str,
    payload: str,
    synthesis: dict[str, object],
    context: dict[str, Any],
    command: str | None,
    timeout_sec: int,
) -> dict[str, object] | None:
    cmd = _normalize_external_command(command)
    if cmd is None:
        return None
    request_payload = {
        "check_type": check_type,
        "payload": payload,
        "context": context,
        "synthesis": synthesis,
    }
    try:
        completed = subprocess.run(
            cmd,
            input=json.dumps(request_payload),
            text=True,
            capture_output=True,
            timeout=max(1, int(timeout_sec)),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "check_type": check_type,
            "passed": False,
            "result": None,
            "error": f"external adapter timed out after {timeout_sec}s",
        }
    except OSError as exc:
        return {
            "check_type": check_type,
            "passed": False,
            "result": None,
            "error": f"external adapter failed to execute: {exc}",
        }

    raw = str(completed.stdout or "").strip()
    if not raw:
        stderr_text = str(completed.stderr or "").strip() or "no output"
        return {
            "check_type": check_type,
            "passed": False,
            "result": None,
            "error": f"external adapter returned empty output ({stderr_text})",
        }
    try:
        decoded = json.loads(raw.splitlines()[-1])
    except json.JSONDecodeError:
        return {
            "check_type": check_type,
            "passed": False,
            "result": None,
            "error": f"invalid external adapter JSON output: {raw[:240]}",
        }
    if not isinstance(decoded, dict):
        return {
            "check_type": check_type,
            "passed": False,
            "result": None,
            "error": "external adapter output must be a JSON object",
        }
    return {
        "check_type": check_type,
        "passed": bool(decoded.get("passed")),
        "result": decoded.get("result"),
        "error": decoded.get("error"),
        "adapter": "external",
    }


def _sample_context(base: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    sampled = dict(base)
    sampled["confidence_score"] = max(
        0.0,
        min(1.0, float(base["confidence_score"]) + rng.uniform(-0.08, 0.08)),
    )
    for key in [
        "change_count",
        "risk_count",
        "experiment_count",
        "finding_count",
        "high_finding_count",
        "critical_finding_count",
    ]:
        sampled[key] = max(0, int(base[key]) + rng.randint(-1, 1))
    sampled["finding_count"] = max(
        sampled["finding_count"],
        sampled["high_finding_count"] + sampled["critical_finding_count"],
    )
    return sampled


def _run_simulation_check(
    payload: str,
    context: dict[str, Any],
    *,
    default_trials: int,
    default_min_pass_rate: float,
    seed: int,
) -> dict[str, object]:
    expression, options = _split_payload(payload)
    if not expression:
        return {
            "check_type": "simulation",
            "expression": "",
            "passed": False,
            "result": None,
            "error": "missing simulation expression",
        }
    trials = max(
        10,
        min(
            2000,
            int(options.get("trials", default_trials)),
        ),
    )
    min_pass_rate = max(
        0.0,
        min(
            1.0,
            float(options.get("min_pass_rate", default_min_pass_rate)),
        ),
    )
    rng = random.Random(seed)
    passed_samples = 0
    try:
        for _ in range(trials):
            sampled_context = _sample_context(context, rng)
            if _safe_eval_expression(expression, sampled_context):
                passed_samples += 1
    except Exception as exc:  # noqa: BLE001
        return {
            "check_type": "simulation",
            "expression": expression,
            "passed": False,
            "result": None,
            "error": str(exc),
        }
    pass_rate = passed_samples / float(trials)
    passed = pass_rate >= min_pass_rate
    return {
        "check_type": "simulation",
        "expression": expression,
        "passed": passed,
        "result": {
            "pass_rate": round(pass_rate, 4),
            "threshold": min_pass_rate,
            "trials": trials,
            "passed_samples": passed_samples,
        },
        "error": None,
    }


def _resolve_data_path(path: str | None, default_data_path: str | None) -> Path | None:
    candidate = str(path or default_data_path or "").strip()
    if not candidate:
        return None
    resolved = Path(candidate)
    return resolved if resolved.exists() else None


def _run_data_reference_check(
    payload: str,
    synthesis: dict[str, object],
    *,
    default_data_path: str | None = None,
) -> dict[str, object]:
    terms_segment, options = _split_payload(payload)
    terms = [str(item).strip().lower() for item in terms_segment.split(",") if str(item).strip()]
    if not terms:
        return {
            "check_type": "data_reference",
            "terms": [],
            "passed": False,
            "result": None,
            "error": "missing data terms",
        }
    source_path = _resolve_data_path(options.get("path"), default_data_path)
    if source_path is None:
        return {
            "check_type": "data_reference",
            "terms": terms,
            "passed": False,
            "result": None,
            "error": "reference dataset path is missing or does not exist",
        }
    try:
        corpus_text = source_path.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        return {
            "check_type": "data_reference",
            "terms": terms,
            "passed": False,
            "result": None,
            "error": str(exc),
        }
    reference_tokens = _tokenize(corpus_text)
    synthesis_tokens = _tokenize(_synthesis_text(synthesis))
    matched_terms: list[str] = []
    for term in terms:
        term_tokens = _tokenize(term)
        if not term_tokens:
            continue
        if term_tokens.issubset(reference_tokens) and term_tokens & synthesis_tokens:
            matched_terms.append(term)
    overlap = len(matched_terms) / float(len(terms))
    min_overlap = max(0.0, min(1.0, float(options.get("min_overlap", 0.5))))
    mode = str(options.get("mode", "fraction")).strip().lower()
    passed = overlap >= min_overlap
    if mode == "all":
        passed = len(matched_terms) == len(terms)
    return {
        "check_type": "data_reference",
        "terms": terms,
        "passed": passed,
        "result": {
            "matched_terms": matched_terms,
            "term_overlap": round(overlap, 4),
            "min_overlap": min_overlap,
            "mode": mode,
            "source_path": str(source_path),
        },
        "error": None,
    }


def run_advanced_verification(
    *,
    constraints: list[str] | None,
    synthesis: dict[str, object],
    findings: list[CriticFinding],
    max_checks: int = 4,
    simulation_trials: int = 200,
    simulation_min_pass_rate: float = 0.7,
    data_reference_path: str | None = None,
    simulation_seed: int = 42,
    external_solver_cmd: str | None = None,
    external_simulation_cmd: str | None = None,
    external_data_cmd: str | None = None,
    external_timeout_sec: int = 8,
) -> dict[str, object]:
    checks = _extract_advanced_checks(constraints, max_checks=max_checks)
    context = _verification_context(synthesis=synthesis, findings=findings)
    results: list[dict[str, object]] = []

    for item in checks:
        kind = str(item.get("kind") or "").strip()
        payload = str(item.get("payload") or "").strip()
        if kind == "solver":
            external = _run_external_adapter(
                check_type="solver",
                payload=payload,
                synthesis=synthesis,
                context=context,
                command=external_solver_cmd,
                timeout_sec=external_timeout_sec,
            )
            results.append(
                external
                if external is not None
                else _run_solver_check(
                    payload,
                    context,
                    backend=_solver_backend(),
                    allow_ast_fallback=True,
                    check_type="solver",
                )
            )
            continue
        if kind == "formal_solver":
            formal_fallback = _parse_bool_env(
                "RIM_ADV_VERIFY_FORMAL_ALLOW_AST_FALLBACK",
                True,
            )
            results.append(
                _run_solver_check(
                    payload,
                    context,
                    backend="z3",
                    allow_ast_fallback=formal_fallback,
                    check_type="formal_solver",
                )
            )
            continue
        if kind == "simulation":
            external = _run_external_adapter(
                check_type="simulation",
                payload=payload,
                synthesis=synthesis,
                context=context,
                command=external_simulation_cmd,
                timeout_sec=external_timeout_sec,
            )
            if external is not None:
                results.append(external)
            else:
                results.append(
                    _run_simulation_check(
                        payload,
                        context,
                        default_trials=max(10, int(simulation_trials)),
                        default_min_pass_rate=max(0.0, min(1.0, float(simulation_min_pass_rate))),
                        seed=int(simulation_seed),
                    )
                )
            continue
        if kind == "data_reference":
            external = _run_external_adapter(
                check_type="data_reference",
                payload=payload,
                synthesis=synthesis,
                context=context,
                command=external_data_cmd,
                timeout_sec=external_timeout_sec,
            )
            if external is not None:
                results.append(external)
            else:
                results.append(
                    _run_data_reference_check(
                        payload,
                        synthesis,
                        default_data_path=data_reference_path,
                    )
                )
            continue

    failed = [item for item in results if not bool(item.get("passed"))]
    errored = [item for item in failed if str(item.get("error") or "").strip()]
    summary = {
        "total_checks": len(results),
        "passed_checks": len(results) - len(failed),
        "failed_checks": len(failed),
        "execution_errors": len(errored),
        "skipped": len(results) == 0,
    }
    return {
        "summary": summary,
        "checks": results,
        "context": context,
    }
