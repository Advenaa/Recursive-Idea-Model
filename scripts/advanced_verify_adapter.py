#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")

ALLOWED_AST_NODES = (
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


def _read_request() -> dict[str, Any]:
    raw = sys.stdin.read().strip()
    if not raw:
        raise ValueError("missing request payload on stdin")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("request payload must be a JSON object")
    return payload


def _print_response(*, passed: bool, result: Any = None, error: str | None = None) -> None:
    print(
        json.dumps(
            {
                "passed": bool(passed),
                "result": result,
                "error": str(error).strip() if error else None,
                "adapter_version": "rim-advanced-adapter-v1",
            }
        )
    )


def _split_payload(payload: str) -> tuple[str, dict[str, str]]:
    segments = [str(item).strip() for item in str(payload).split("|") if str(item).strip()]
    if not segments:
        return "", {}
    expression = segments[0]
    options: dict[str, str] = {}
    for part in segments[1:]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = str(key).strip().lower()
        value = str(value).strip()
        if not key or not value:
            continue
        options[key] = value
    return expression, options


def _tokenize(text: str) -> set[str]:
    return {item.group(0).lower() for item in TOKEN_RE.finditer(str(text))}


def _synthesis_text(synthesis: dict[str, Any]) -> str:
    chunks = [str(synthesis.get("synthesized_idea") or "").strip()]
    chunks.extend(str(item).strip() for item in list(synthesis.get("changes_summary") or []) if str(item).strip())
    chunks.extend(str(item).strip() for item in list(synthesis.get("next_experiments") or []) if str(item).strip())
    return " ".join(chunk for chunk in chunks if chunk)


def _validate_ast(node: ast.AST, allowed_names: set[str]) -> None:
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            raise ValueError("function calls are not allowed")
        if isinstance(child, ast.Attribute):
            raise ValueError("attribute access is not allowed")
        if isinstance(child, ast.Subscript):
            raise ValueError("subscript access is not allowed")
        if not isinstance(child, ALLOWED_AST_NODES):
            raise ValueError(f"unsupported syntax: {type(child).__name__}")
        if isinstance(child, ast.Name) and child.id not in allowed_names:
            raise ValueError(f"unknown variable '{child.id}'")


def _safe_eval(expression: str, context: dict[str, Any]) -> bool:
    parsed = ast.parse(expression, mode="eval")
    _validate_ast(parsed, set(context.keys()))
    compiled = compile(parsed, "<advanced-verify-adapter>", "eval")
    value = eval(compiled, {"__builtins__": {}}, dict(context))  # noqa: S307
    return bool(value)


def _sample_context(base: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    sampled = dict(base)
    sampled["confidence_score"] = max(
        0.0,
        min(1.0, float(base.get("confidence_score", 0.0)) + rng.uniform(-0.08, 0.08)),
    )
    for key in [
        "change_count",
        "risk_count",
        "experiment_count",
        "finding_count",
        "high_finding_count",
        "critical_finding_count",
    ]:
        sampled[key] = max(0, int(base.get(key, 0)) + rng.randint(-1, 1))
    sampled["finding_count"] = max(
        int(sampled["finding_count"]),
        int(sampled["high_finding_count"]) + int(sampled["critical_finding_count"]),
    )
    return sampled


def _run_solver(payload: str, context: dict[str, Any]) -> tuple[bool, Any, str | None]:
    expression = str(payload).strip()
    if not expression:
        return False, None, "missing solver expression"
    try:
        passed = _safe_eval(expression, context)
        return passed, {"expression": expression}, None
    except Exception as exc:  # noqa: BLE001
        return False, None, str(exc)


def _run_simulation(payload: str, context: dict[str, Any]) -> tuple[bool, Any, str | None]:
    expression, options = _split_payload(payload)
    if not expression:
        return False, None, "missing simulation expression"
    default_trials = int(os.getenv("RIM_ADV_VERIFY_ADAPTER_SIM_TRIALS", "200"))
    default_min_pass_rate = float(os.getenv("RIM_ADV_VERIFY_ADAPTER_SIM_MIN_PASS_RATE", "0.7"))
    seed = int(os.getenv("RIM_ADV_VERIFY_ADAPTER_SEED", "42"))
    trials = max(10, min(2000, int(options.get("trials", default_trials))))
    min_pass_rate = max(0.0, min(1.0, float(options.get("min_pass_rate", default_min_pass_rate))))
    rng = random.Random(seed)
    passed_samples = 0
    try:
        for _ in range(trials):
            if _safe_eval(expression, _sample_context(context, rng)):
                passed_samples += 1
    except Exception as exc:  # noqa: BLE001
        return False, None, str(exc)
    pass_rate = passed_samples / float(trials)
    return (
        pass_rate >= min_pass_rate,
        {
            "expression": expression,
            "trials": trials,
            "passed_samples": passed_samples,
            "pass_rate": round(pass_rate, 4),
            "min_pass_rate": min_pass_rate,
        },
        None,
    )


def _run_data_reference(payload: str, synthesis: dict[str, Any]) -> tuple[bool, Any, str | None]:
    terms_segment, options = _split_payload(payload)
    terms = [str(item).strip().lower() for item in terms_segment.split(",") if str(item).strip()]
    if not terms:
        return False, None, "missing data terms"
    default_path = os.getenv("RIM_ADV_VERIFY_ADAPTER_DATA_PATH", "")
    source_path = Path(str(options.get("path", default_path)).strip()) if str(options.get("path", default_path)).strip() else None
    if source_path is None or not source_path.exists():
        return False, None, "reference dataset path is missing or does not exist"
    try:
        corpus = source_path.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        return False, None, str(exc)
    reference_tokens = _tokenize(corpus)
    synthesis_tokens = _tokenize(_synthesis_text(synthesis))
    matched: list[str] = []
    for term in terms:
        term_tokens = _tokenize(term)
        if term_tokens and term_tokens.issubset(reference_tokens) and (term_tokens & synthesis_tokens):
            matched.append(term)
    overlap = len(matched) / float(len(terms))
    min_overlap = max(0.0, min(1.0, float(options.get("min_overlap", 0.5))))
    mode = str(options.get("mode", "fraction")).strip().lower()
    passed = overlap >= min_overlap
    if mode == "all":
        passed = len(matched) == len(terms)
    return (
        passed,
        {
            "source_path": str(source_path),
            "terms": terms,
            "matched_terms": matched,
            "term_overlap": round(overlap, 4),
            "min_overlap": min_overlap,
            "mode": mode,
        },
        None,
    )


def main() -> int:
    try:
        request = _read_request()
        check_type = str(request.get("check_type") or "").strip().lower()
        payload = str(request.get("payload") or "").strip()
        context = request.get("context") if isinstance(request.get("context"), dict) else {}
        synthesis = request.get("synthesis") if isinstance(request.get("synthesis"), dict) else {}
        if check_type == "solver":
            passed, result, error = _run_solver(payload, context)
            _print_response(passed=passed, result=result, error=error)
            return 0
        if check_type == "simulation":
            passed, result, error = _run_simulation(payload, context)
            _print_response(passed=passed, result=result, error=error)
            return 0
        if check_type == "data_reference":
            passed, result, error = _run_data_reference(payload, synthesis)
            _print_response(passed=passed, result=result, error=error)
            return 0
        _print_response(passed=False, result=None, error=f"unsupported check_type '{check_type}'")
        return 0
    except Exception as exc:  # noqa: BLE001
        _print_response(passed=False, result=None, error=str(exc))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
