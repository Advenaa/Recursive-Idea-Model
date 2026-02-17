from __future__ import annotations

import ast
import json
import os
import subprocess
import random
import re
import shlex
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from rim.core.schemas import CriticFinding

_SOLVER_PREFIXES = ("solver:", "solve:")
_FORMAL_SOLVER_PREFIXES = ("formal:", "theorem:", "constraint:")
_SIM_PREFIXES = ("simulate:", "simulation:")
_DATA_PREFIXES = ("data:", "dataset:")
_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")
_FORMAL_COUNT_KEYS = (
    "change_count",
    "risk_count",
    "experiment_count",
    "finding_count",
    "high_finding_count",
    "critical_finding_count",
)

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
        if matched is None:
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
        if hasattr(value, "sort"):
            return value
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


def _parse_int_env(name: str, default: int, *, lower: int, upper: int) -> int:
    raw = os.getenv(name)
    try:
        value = int(str(raw)) if raw is not None else int(default)
    except (TypeError, ValueError):
        value = int(default)
    return max(lower, min(upper, value))


def _split_assumptions(raw: str | None) -> list[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    return [item for item in [part.strip() for part in text.split(";")] if item]


def _z3_value_to_python(value: Any, z3: Any) -> object:  # noqa: ANN401
    try:
        if z3.is_true(value):
            return True
        if z3.is_false(value):
            return False
    except Exception:  # noqa: BLE001
        pass
    text = str(value)
    if "/" in text:
        left, _, right = text.partition("/")
        try:
            return float(left) / float(right)
        except (TypeError, ValueError, ZeroDivisionError):
            return text
    try:
        return int(text)
    except (TypeError, ValueError):
        pass
    try:
        return float(text)
    except (TypeError, ValueError):
        return text


def _formal_context_bounds(
    *,
    upper_bound: int,
) -> dict[str, tuple[float, float]]:
    bounds: dict[str, tuple[float, float]] = {
        "confidence_score": (0.0, 1.0),
    }
    for key in _FORMAL_COUNT_KEYS:
        bounds[key] = (0.0, float(max(1, int(upper_bound))))
    return bounds


def _context_fallback_formal_check(
    *,
    expression: str,
    assumptions: list[str],
    context: dict[str, Any],
    mode: str,
) -> dict[str, object]:
    try:
        assumptions_hold = all(_safe_eval_expression(item, context) for item in assumptions)
        expression_holds = _safe_eval_expression(expression, context)
    except Exception as exc:  # noqa: BLE001
        return {
            "check_type": "formal_solver",
            "expression": expression,
            "mode": mode,
            "assumptions": assumptions,
            "passed": False,
            "result": None,
            "error": str(exc),
        }
    if mode == "prove":
        passed = (not assumptions_hold) or expression_holds
    elif mode == "refute":
        passed = assumptions_hold and (not expression_holds)
    else:
        passed = assumptions_hold and expression_holds
    return {
        "check_type": "formal_solver",
        "expression": expression,
        "mode": mode,
        "assumptions": assumptions,
        "passed": passed,
        "result": {
            "backend": "ast_context_fallback",
            "assumptions_hold": assumptions_hold,
            "expression_holds": expression_holds,
            "mode": mode,
        },
        "error": None,
    }


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


def _run_formal_solver_check(
    payload: str,
    context: dict[str, Any],
    *,
    allow_ast_fallback: bool,
) -> dict[str, object]:
    expression, options = _split_payload(payload)
    expression = str(expression).strip()
    if not expression:
        return {
            "check_type": "formal_solver",
            "expression": "",
            "mode": "prove",
            "assumptions": [],
            "passed": False,
            "result": None,
            "error": "missing formal solver expression",
        }

    mode = str(options.get("mode", "prove")).strip().lower()
    if mode not in {"prove", "satisfiable", "refute"}:
        return {
            "check_type": "formal_solver",
            "expression": expression,
            "mode": mode,
            "assumptions": [],
            "passed": False,
            "result": None,
            "error": "formal mode must be one of: prove, satisfiable, refute",
        }
    assumptions = _split_assumptions(options.get("assume"))
    upper_bound = _parse_int_env(
        "RIM_ADV_VERIFY_FORMAL_MAX_COUNT",
        200,
        lower=1,
        upper=100000,
    )

    try:
        import z3  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        if allow_ast_fallback:
            result = _context_fallback_formal_check(
                expression=expression,
                assumptions=assumptions,
                context=context,
                mode=mode,
            )
            if result.get("error") is None and isinstance(result.get("result"), dict):
                result_payload = dict(result["result"])
                result_payload["fallback_reason"] = f"z3 unavailable: {exc}"
                result["result"] = result_payload
            return result
        return {
            "check_type": "formal_solver",
            "expression": expression,
            "mode": mode,
            "assumptions": assumptions,
            "passed": False,
            "result": None,
            "error": f"z3 backend unavailable: {exc}",
        }

    symbolic_context: dict[str, Any] = {
        key: z3.Real(key) if key != "confidence_score" else z3.Real(key)
        for key in context.keys()
    }
    bounds = _formal_context_bounds(upper_bound=upper_bound)
    constraints: list[Any] = []
    for key, symbol in symbolic_context.items():
        if key not in bounds:
            continue
        lower, upper = bounds[key]
        constraints.append(symbol >= float(lower))
        constraints.append(symbol <= float(upper))
    if {
        "finding_count",
        "high_finding_count",
        "critical_finding_count",
    }.issubset(symbolic_context.keys()):
        constraints.append(
            symbolic_context["finding_count"]
            >= symbolic_context["high_finding_count"] + symbolic_context["critical_finding_count"]
        )

    try:
        parsed_expression = _z3_from_ast(
            ast.parse(expression, mode="eval"),
            symbolic_context,
            z3,
        )
        parsed_assumptions = [
            _z3_from_ast(
                ast.parse(item, mode="eval"),
                symbolic_context,
                z3,
            )
            for item in assumptions
        ]
    except Exception as exc:  # noqa: BLE001
        if allow_ast_fallback:
            fallback = _context_fallback_formal_check(
                expression=expression,
                assumptions=assumptions,
                context=context,
                mode=mode,
            )
            if fallback.get("error") is None and isinstance(fallback.get("result"), dict):
                payload_result = dict(fallback["result"])
                payload_result["fallback_reason"] = f"formal parse failure: {exc}"
                fallback["result"] = payload_result
            return fallback
        return {
            "check_type": "formal_solver",
            "expression": expression,
            "mode": mode,
            "assumptions": assumptions,
            "passed": False,
            "result": None,
            "error": str(exc),
        }

    solver = z3.Solver()
    solver.add(*constraints)
    solver.add(*parsed_assumptions)
    if mode == "prove":
        solver.add(z3.Not(parsed_expression))
        solver_status = solver.check()
        passed = solver_status == z3.unsat
    elif mode == "refute":
        solver.add(parsed_expression)
        solver_status = solver.check()
        passed = solver_status == z3.unsat
    else:  # satisfiable
        solver.add(parsed_expression)
        solver_status = solver.check()
        passed = solver_status == z3.sat

    result_payload: dict[str, Any] = {
        "backend": "z3_formal",
        "mode": mode,
        "solver_status": str(solver_status),
        "assumption_count": len(assumptions),
        "max_count_bound": upper_bound,
    }
    if not passed and solver_status == z3.sat:
        model = solver.model()
        counterexample: dict[str, object] = {}
        for key, symbol in symbolic_context.items():
            try:
                counterexample[key] = _z3_value_to_python(
                    model.eval(symbol, model_completion=True),
                    z3,
                )
            except Exception:  # noqa: BLE001
                continue
        if counterexample:
            result_payload["counterexample"] = counterexample
    return {
        "check_type": "formal_solver",
        "expression": expression,
        "mode": mode,
        "assumptions": assumptions,
        "passed": bool(passed),
        "result": result_payload,
        "error": None,
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


def _is_http_url(value: str | None) -> bool:
    text = str(value or "").strip().lower()
    return text.startswith("https://") or text.startswith("http://")


def _http_host(value: str | None) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        return ""
    if "://" not in normalized:
        normalized = f"https://{normalized}"
    try:
        parsed = urlparse(normalized)
    except Exception:  # noqa: BLE001
        return ""
    return str(parsed.hostname or "").strip().lower().rstrip(".")


def _normalize_http_allowlist(value: object) -> list[str]:
    if value is None:
        return []
    raw_items: list[object]
    if isinstance(value, str):
        raw_items = [item for item in str(value).split(",")]
    elif isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    else:
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        host = _http_host(item)
        if not host or host in seen:
            continue
        seen.add(host)
        normalized.append(host)
    return normalized


def _is_allowed_http_source(source_url: str, allowed_hosts: list[str]) -> bool:
    if not allowed_hosts:
        return True
    host = _http_host(source_url)
    if not host:
        return False
    for allowed in allowed_hosts:
        normalized_allowed = _http_host(allowed)
        if not normalized_allowed:
            continue
        if host == normalized_allowed:
            return True
        if host.endswith(f".{normalized_allowed}"):
            return True
    return False


def _fetch_http_text(
    *,
    source_url: str,
    timeout_sec: int,
    max_bytes: int,
) -> tuple[str | None, str | None]:
    normalized_url = str(source_url or "").strip()
    if not normalized_url:
        return None, "missing source URL"
    try:
        request = Request(
            normalized_url,
            headers={"User-Agent": "RIM-AdvancedVerifier/1.0"},
        )
        with urlopen(request, timeout=max(1, int(timeout_sec))) as response:
            raw = response.read(max(1, int(max_bytes)) + 1)
    except (URLError, TimeoutError, OSError, ValueError) as exc:
        return None, str(exc)
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)
    if len(raw) > int(max_bytes):
        return None, f"HTTP data exceeds max bytes ({max_bytes})"
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError:
        decoded = raw.decode("utf-8", errors="replace")
    return decoded, None


def _run_data_reference_check(
    payload: str,
    synthesis: dict[str, object],
    *,
    default_data_path: str | None = None,
    allow_http: bool = False,
    http_timeout_sec: int = 5,
    http_max_bytes: int = 300_000,
    http_allowed_hosts: list[str] | None = None,
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
    path_option = str(options.get("path") or "").strip()
    url_option = str(options.get("url") or "").strip()
    source_url = url_option
    if not source_url and _is_http_url(path_option):
        source_url = path_option
    if not source_url and _is_http_url(default_data_path):
        source_url = str(default_data_path or "").strip()

    source_path: Path | None = None
    corpus_text = ""
    source_kind = "path"
    source_value = ""
    allowed_hosts = _normalize_http_allowlist(http_allowed_hosts)
    if source_url:
        if not allow_http:
            return {
                "check_type": "data_reference",
                "terms": terms,
                "passed": False,
                "result": None,
                "error": "HTTP data references are disabled (set allow_http_data_reference).",
            }
        if allowed_hosts and not _is_allowed_http_source(source_url, allowed_hosts):
            return {
                "check_type": "data_reference",
                "terms": terms,
                "passed": False,
                "result": None,
                "error": (
                    f"HTTP data reference host '{_http_host(source_url) or source_url}' "
                    "is not in the allowed host list."
                ),
            }
        corpus_text, fetch_error = _fetch_http_text(
            source_url=source_url,
            timeout_sec=max(1, int(http_timeout_sec)),
            max_bytes=max(1, int(http_max_bytes)),
        )
        if fetch_error is not None or corpus_text is None:
            return {
                "check_type": "data_reference",
                "terms": terms,
                "passed": False,
                "result": None,
                "error": str(fetch_error or "failed to fetch HTTP data reference"),
            }
        source_kind = "url"
        source_value = source_url
    else:
        source_path = _resolve_data_path(path_option, default_data_path)
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
        source_value = str(source_path)
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
            "source_kind": source_kind,
            "source": source_value,
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
    allow_http_data_reference: bool = False,
    http_data_timeout_sec: int = 5,
    http_data_max_bytes: int = 300_000,
    http_data_allowed_hosts: list[str] | None = None,
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
                _run_formal_solver_check(
                    payload,
                    context,
                    allow_ast_fallback=formal_fallback,
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
                        allow_http=allow_http_data_reference,
                        http_timeout_sec=http_data_timeout_sec,
                        http_max_bytes=http_data_max_bytes,
                        http_allowed_hosts=http_data_allowed_hosts,
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
