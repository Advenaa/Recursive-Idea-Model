from __future__ import annotations

import ast
from typing import Any

from rim.core.schemas import CriticFinding

_CHECK_PREFIXES = ("python:", "py:", "assert:")

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


def _extract_executable_checks(constraints: list[str] | None, max_checks: int) -> list[str]:
    checks: list[str] = []
    for item in list(constraints or []):
        text = str(item).strip()
        if not text:
            continue
        lowered = text.lower()
        matched = None
        for prefix in _CHECK_PREFIXES:
            if lowered.startswith(prefix):
                matched = text[len(prefix) :].strip()
                break
        if matched is None:
            continue
        if matched:
            checks.append(matched)
        if len(checks) >= max(1, int(max_checks)):
            break
    return checks


def _validate_ast(node: ast.AST, allowed_names: set[str]) -> None:
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            raise ValueError("Function calls are not allowed in executable checks.")
        if isinstance(child, ast.Attribute):
            raise ValueError("Attribute access is not allowed in executable checks.")
        if isinstance(child, ast.Subscript):
            raise ValueError("Subscript access is not allowed in executable checks.")
        if not isinstance(child, _ALLOWED_AST_NODES):
            raise ValueError(f"Unsupported syntax in executable check: {type(child).__name__}")
        if isinstance(child, ast.Name) and child.id not in allowed_names:
            raise ValueError(f"Unknown variable '{child.id}' in executable check.")


def _safe_eval_expression(expression: str, context: dict[str, Any]) -> Any:
    parsed = ast.parse(expression, mode="eval")
    _validate_ast(parsed, set(context.keys()))
    compiled = compile(parsed, "<rim-check>", "eval")
    return eval(compiled, {"__builtins__": {}}, dict(context))  # noqa: S307


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


def run_executable_verification(
    *,
    constraints: list[str] | None,
    synthesis: dict[str, object],
    findings: list[CriticFinding],
    max_checks: int = 5,
) -> dict[str, object]:
    checks = _extract_executable_checks(constraints, max_checks=max_checks)
    context = _verification_context(synthesis=synthesis, findings=findings)
    results: list[dict[str, object]] = []

    for expression in checks:
        try:
            value = _safe_eval_expression(expression, context)
            passed = bool(value)
            results.append(
                {
                    "check_type": "python_expression",
                    "expression": expression,
                    "passed": passed,
                    "result": value,
                    "error": None,
                }
            )
        except Exception as exc:  # noqa: BLE001
            results.append(
                {
                    "check_type": "python_expression",
                    "expression": expression,
                    "passed": False,
                    "result": None,
                    "error": str(exc),
                }
            )

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
