from __future__ import annotations

import ast
import json
import subprocess
import sys
from typing import Any

from rim.core.schemas import CriticFinding

_CHECK_PREFIXES = ("python:", "py:", "assert:")
_EXEC_PREFIXES = ("python_exec:", "exec:")

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


def _extract_executable_checks(constraints: list[str] | None, max_checks: int) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    for item in list(constraints or []):
        text = str(item).strip()
        if not text:
            continue
        lowered = text.lower()
        matched: dict[str, str] | None = None
        for prefix in _CHECK_PREFIXES:
            if lowered.startswith(prefix):
                matched = {
                    "kind": "python_expression",
                    "payload": text[len(prefix) :].strip(),
                }
                break
        if matched is None:
            for prefix in _EXEC_PREFIXES:
                if lowered.startswith(prefix):
                    matched = {
                        "kind": "python_exec",
                        "payload": text[len(prefix) :].strip(),
                    }
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


def _run_python_exec_check(
    script: str,
    context: dict[str, Any],
    timeout_sec: int,
) -> tuple[bool, str | None, Any]:
    wrapper = (
        "import json,sys\n"
        "context=json.loads(sys.argv[1])\n"
        "user_script=sys.argv[2]\n"
        "safe_builtins={'len':len,'min':min,'max':max,'sum':sum,'abs':abs,'all':all,'any':any}\n"
        "scope={'context':context,'passed':False,'detail':''}\n"
        "try:\n"
        "    exec(user_script, {'__builtins__': safe_builtins}, scope)\n"
        "    passed=bool(scope.get('passed', False))\n"
        "    detail=scope.get('detail', '')\n"
        "    print(json.dumps({'passed': passed, 'detail': detail}))\n"
        "except Exception as exc:\n"
        "    print(json.dumps({'passed': False, 'error': str(exc)}))\n"
        "    sys.exit(3)\n"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", wrapper, json.dumps(context), script],
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout_sec)),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, "python_exec check timed out", None

    output = (completed.stdout or "").strip()
    if not output:
        return False, (completed.stderr or "python_exec check produced no output").strip(), None
    try:
        payload = json.loads(output.splitlines()[-1])
    except json.JSONDecodeError:
        return False, f"invalid python_exec output: {output[:200]}", None

    passed = bool(payload.get("passed"))
    if completed.returncode == 0:
        return passed, None if passed else "python_exec check failed", payload.get("detail")
    error = str(payload.get("error") or payload.get("detail") or "python_exec runtime error").strip()
    return False, error, payload.get("detail")


def run_executable_verification(
    *,
    constraints: list[str] | None,
    synthesis: dict[str, object],
    findings: list[CriticFinding],
    max_checks: int = 5,
    enable_python_exec: bool = False,
    python_exec_timeout_sec: int = 2,
) -> dict[str, object]:
    checks = _extract_executable_checks(constraints, max_checks=max_checks)
    context = _verification_context(synthesis=synthesis, findings=findings)
    results: list[dict[str, object]] = []

    for check in checks:
        kind = str(check.get("kind") or "python_expression")
        payload = str(check.get("payload") or "").strip()
        if not payload:
            continue
        if kind == "python_exec":
            if not enable_python_exec:
                results.append(
                    {
                        "check_type": "python_exec",
                        "expression": payload,
                        "passed": False,
                        "result": None,
                        "error": "python_exec checks are disabled",
                    }
                )
                continue
            passed, error, detail = _run_python_exec_check(
                payload,
                context,
                timeout_sec=python_exec_timeout_sec,
            )
            results.append(
                {
                    "check_type": "python_exec",
                    "expression": payload,
                    "passed": passed,
                    "result": detail,
                    "error": error,
                }
            )
            continue

        try:
            value = _safe_eval_expression(payload, context)
            passed = bool(value)
            results.append(
                {
                    "check_type": "python_expression",
                    "expression": payload,
                    "passed": passed,
                    "result": value,
                    "error": None,
                }
            )
        except Exception as exc:  # noqa: BLE001
            results.append(
                {
                    "check_type": "python_expression",
                    "expression": payload,
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
