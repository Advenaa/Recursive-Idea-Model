from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from rim.core.schemas import CriticFinding, DecompositionNode
from rim.storage.db import get_connection, init_db

SEVERITY_RANK = {
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}
TERMINAL_RUN_STATUSES = {"completed", "failed", "partial", "canceled"}
RETRYABLE_RUN_STATUSES = {"failed", "partial", "canceled"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_severity(value: str | None) -> str:
    parsed = str(value or "medium").strip().lower()
    if parsed in SEVERITY_RANK:
        return parsed
    return "medium"


def _feedback_delta(entry_type: str, verdict: str) -> float:
    if verdict == "accept":
        boosts = {
            "insight": 0.12,
            "pattern": 0.10,
            "failure": 0.05,
            "feedback": 0.08,
        }
        return boosts.get(entry_type, 0.08)

    penalties = {
        "insight": -0.20,
        "pattern": -0.12,
        # Negative feedback increases failure/risk memory relevance.
        "failure": 0.10,
        "feedback": 0.05,
    }
    return penalties.get(entry_type, -0.10)


def _parse_error_summary(value: str | None) -> tuple[str | None, dict | None]:
    raw = str(value or "").strip()
    if not raw:
        return None, None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return raw, None
    if not isinstance(payload, dict):
        return raw, None
    message = payload.get("message")
    if not isinstance(message, str) or not message.strip():
        message = raw
    return message.strip(), payload


def _parse_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


class RunRepository:
    def __init__(self, db_path: str | Path | None = None) -> None:
        self.conn = get_connection(db_path)
        init_db(self.conn)

    def healthcheck(self) -> bool:
        try:
            self.conn.execute("SELECT 1")
            return True
        except sqlite3.Error:
            return False

    def create_run(self, run_id: str, mode: str, input_idea: str) -> None:
        self.create_run_with_request(
            run_id=run_id,
            mode=mode,
            input_idea=input_idea,
            request_json=None,
            status="running",
        )

    def create_run_with_request(
        self,
        run_id: str,
        mode: str,
        input_idea: str,
        request_json: str | None,
        status: str = "queued",
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO runs (id, mode, input_idea, request_json, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (run_id, mode, input_idea, request_json, status, _utc_now()),
        )
        self.conn.commit()

    def mark_run_status(
        self,
        run_id: str,
        status: str,
        confidence_score: float | None = None,
        error_summary: str | None = None,
    ) -> None:
        completed_at = _utc_now() if status in TERMINAL_RUN_STATUSES else None
        self.conn.execute(
            """
            UPDATE runs
            SET status = ?, completed_at = ?, confidence_score = ?, error_summary = ?
            WHERE id = ?
            """,
            (status, completed_at, confidence_score, error_summary, run_id),
        )
        self.conn.commit()

    def log_stage(
        self,
        run_id: str,
        stage: str,
        status: str,
        provider: str | None = None,
        latency_ms: int | None = None,
        meta: dict | None = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO stage_logs (id, run_id, stage, provider, latency_ms, status, meta_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid4()),
                run_id,
                stage,
                provider,
                latency_ms,
                status,
                json.dumps(meta or {}),
                _utc_now(),
            ),
        )
        self.conn.commit()

    def save_nodes(self, run_id: str, nodes: list[DecompositionNode]) -> None:
        self.conn.executemany(
            """
            INSERT INTO nodes (id, run_id, parent_node_id, depth, component_text, node_type, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    node.id,
                    run_id,
                    node.parent_node_id,
                    node.depth,
                    node.component_text,
                    node.node_type,
                    node.confidence,
                )
                for node in nodes
            ],
        )
        self.conn.commit()

    def save_findings(self, run_id: str, findings: list[CriticFinding]) -> None:
        self.conn.executemany(
            """
            INSERT INTO critic_findings
            (id, run_id, node_id, critic_type, issue, severity, confidence, suggested_fix, provider)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    finding.id,
                    run_id,
                    finding.node_id,
                    finding.critic_type,
                    finding.issue,
                    finding.severity,
                    finding.confidence,
                    finding.suggested_fix,
                    finding.provider,
                )
                for finding in findings
            ],
        )
        self.conn.commit()

    def save_synthesis(
        self,
        run_id: str,
        synthesized_idea: str,
        changes_summary: list[str],
        residual_risks: list[str],
        next_experiments: list[str],
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO synthesis_outputs
            (id, run_id, synthesized_idea, changes_summary_json, residual_risks_json, next_experiments_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid4()),
                run_id,
                synthesized_idea,
                json.dumps(changes_summary),
                json.dumps(residual_risks),
                json.dumps(next_experiments),
            ),
        )
        self.conn.commit()

    def save_memory_entries(self, run_id: str, entries: list[dict]) -> None:
        if not entries:
            return
        self.conn.executemany(
            """
            INSERT INTO memory_entries (id, run_id, entry_type, entry_text, domain, severity, score, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    str(uuid4()),
                    run_id,
                    str(item.get("entry_type", "insight")),
                    str(item.get("entry_text", "")).strip(),
                    str(item.get("domain", "")).strip() or None,
                    _normalize_severity(item.get("severity")),
                    float(item.get("score", 0.0)),
                    _utc_now(),
                )
                for item in entries
                if str(item.get("entry_text", "")).strip()
            ],
        )
        self.conn.commit()

    def get_memory_context(
        self,
        limit: int = 8,
        domain: str | None = None,
        max_age_days: int | None = None,
        min_severity: str = "low",
    ) -> list[str]:
        where: list[str] = [
            "entry_text IS NOT NULL",
            "TRIM(entry_text) != ''",
            "COALESCE(score, 0.0) > 0.0",
        ]
        params: list[object] = []

        min_rank = SEVERITY_RANK.get(_normalize_severity(min_severity), 1)
        where.append(
            """
            (CASE LOWER(COALESCE(severity, 'medium'))
                WHEN 'low' THEN 1
                WHEN 'medium' THEN 2
                WHEN 'high' THEN 3
                WHEN 'critical' THEN 4
                ELSE 2
            END) >= ?
            """.strip()
        )
        params.append(min_rank)

        if domain and str(domain).strip():
            where.append(
                "(LOWER(TRIM(domain)) = LOWER(TRIM(?)) OR domain IS NULL OR TRIM(domain) = '')"
            )
            params.append(str(domain).strip())

        if max_age_days is not None and max_age_days > 0:
            where.append("julianday(created_at) >= julianday('now', ?)")
            params.append(f"-{int(max_age_days)} days")

        query = f"""
            SELECT entry_text
            FROM memory_entries
            WHERE {' AND '.join(where)}
            ORDER BY COALESCE(score, 0.0) DESC, created_at DESC
            LIMIT ?
        """
        params.append(limit)
        rows = self.conn.execute(query, params).fetchall()
        return [str(row["entry_text"]) for row in rows]

    def get_run(self, run_id: str) -> dict | None:
        run_row = self.conn.execute(
            "SELECT * FROM runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        if run_row is None:
            return None

        node_rows = self.conn.execute(
            "SELECT * FROM nodes WHERE run_id = ? ORDER BY depth ASC",
            (run_id,),
        ).fetchall()
        finding_rows = self.conn.execute(
            "SELECT * FROM critic_findings WHERE run_id = ?",
            (run_id,),
        ).fetchall()
        synthesis_row = self.conn.execute(
            "SELECT * FROM synthesis_outputs WHERE run_id = ?",
            (run_id,),
        ).fetchone()

        result_payload = None
        if synthesis_row is not None:
            result_payload = {
                "run_id": run_row["id"],
                "mode": run_row["mode"],
                "input_idea": run_row["input_idea"],
                "decomposition": [dict(row) for row in node_rows],
                "critic_findings": [dict(row) for row in finding_rows],
                "synthesized_idea": synthesis_row["synthesized_idea"],
                "changes_summary": json.loads(synthesis_row["changes_summary_json"]),
                "residual_risks": json.loads(synthesis_row["residual_risks_json"]),
                "next_experiments": json.loads(synthesis_row["next_experiments_json"]),
                "confidence_score": run_row["confidence_score"] or 0.0,
            }

        error_summary, error_payload = _parse_error_summary(run_row["error_summary"])
        return {
            "run_id": run_row["id"],
            "status": run_row["status"],
            "error_summary": error_summary,
            "error": error_payload,
            "result": result_payload,
        }

    def get_run_request(self, run_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT request_json FROM runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        payload = row["request_json"]
        if not payload:
            return None
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError:
            return None
        return decoded if isinstance(decoded, dict) else None

    def reset_run_for_retry(self, run_id: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            return False
        self.conn.execute("DELETE FROM nodes WHERE run_id = ?", (run_id,))
        self.conn.execute("DELETE FROM critic_findings WHERE run_id = ?", (run_id,))
        self.conn.execute("DELETE FROM synthesis_outputs WHERE run_id = ?", (run_id,))
        self.conn.execute("DELETE FROM memory_entries WHERE run_id = ?", (run_id,))
        self.conn.execute("DELETE FROM run_feedback WHERE run_id = ?", (run_id,))
        self.conn.execute("DELETE FROM stage_logs WHERE run_id = ?", (run_id,))
        self.conn.execute(
            """
            UPDATE runs
            SET status = 'queued', completed_at = NULL, confidence_score = NULL, error_summary = NULL
            WHERE id = ?
            """,
            (run_id,),
        )
        self.conn.commit()
        return True

    def get_run_domain(self, run_id: str) -> str | None:
        payload = self.get_run_request(run_id)
        if payload is None:
            return None
        domain = payload.get("domain")
        if domain is None:
            return None
        parsed = str(domain).strip()
        return parsed or None

    def get_pending_runs(self, limit: int = 200, *, include_running: bool = True) -> list[str]:
        statuses = ["queued"]
        if include_running:
            statuses.append("running")
        placeholders = ",".join("?" for _ in statuses)
        rows = self.conn.execute(
            f"""
            SELECT id
            FROM runs
            WHERE status IN ({placeholders})
            ORDER BY created_at ASC
            LIMIT ?
            """,
            (*statuses, limit),
        ).fetchall()
        return [str(row["id"]) for row in rows]

    def list_runs(
        self,
        *,
        status: str | None = None,
        mode: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict]:
        limit = max(1, min(int(limit), 200))
        offset = max(0, int(offset))
        where: list[str] = []
        params: list[object] = []

        if status:
            where.append("status = ?")
            params.append(str(status).strip().lower())
        if mode:
            where.append("mode = ?")
            params.append(str(mode).strip().lower())

        sql = """
            SELECT
                id,
                mode,
                input_idea,
                status,
                created_at,
                completed_at,
                confidence_score,
                error_summary
            FROM runs
        """
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        rows = self.conn.execute(sql, params).fetchall()
        items: list[dict] = []
        for row in rows:
            item = dict(row)
            message, _error_payload = _parse_error_summary(item.get("error_summary"))
            item["error_summary"] = message
            items.append(item)
        return items

    def submit_run_feedback(
        self,
        run_id: str,
        verdict: str,
        notes: str | None = None,
    ) -> dict:
        run_exists = self.conn.execute(
            "SELECT 1 FROM runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        if run_exists is None:
            raise ValueError("Run not found")

        verdict = str(verdict).strip().lower()
        if verdict not in {"accept", "reject"}:
            raise ValueError("Feedback verdict must be 'accept' or 'reject'")

        created_at = _utc_now()
        self.conn.execute(
            """
            INSERT INTO run_feedback (id, run_id, verdict, notes, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (str(uuid4()), run_id, verdict, notes, created_at),
        )

        updated = 0
        rows = self.conn.execute(
            "SELECT id, entry_type, score FROM memory_entries WHERE run_id = ?",
            (run_id,),
        ).fetchall()
        updates: list[tuple[float, str]] = []
        for row in rows:
            entry_type = str(row["entry_type"] or "insight")
            current = float(row["score"] or 0.0)
            adjusted = max(0.0, min(1.0, current + _feedback_delta(entry_type, verdict)))
            updates.append((adjusted, str(row["id"])))
        if updates:
            self.conn.executemany(
                "UPDATE memory_entries SET score = ? WHERE id = ?",
                updates,
            )
            updated += len(updates)

        note = str(notes or "").strip()
        if note:
            self.save_memory_entries(
                run_id=run_id,
                entries=[
                    {
                        "entry_type": "feedback",
                        "entry_text": f"User feedback ({verdict}): {note}",
                        "domain": self.get_run_domain(run_id),
                        "severity": "high" if verdict == "reject" else "medium",
                        "score": 0.9 if verdict == "reject" else 0.75,
                    }
                ],
            )
            updated += 1
        else:
            self.conn.commit()

        return {
            "run_id": run_id,
            "verdict": verdict,
            "notes": note or None,
            "updated_memory_entries": updated,
            "created_at": created_at,
        }

    def get_stage_logs(self, run_id: str) -> list[dict]:
        rows = self.conn.execute(
            """
            SELECT stage, provider, latency_ms, status, meta_json, created_at
            FROM stage_logs
            WHERE run_id = ?
            ORDER BY created_at ASC
            """,
            (run_id,),
        ).fetchall()
        logs: list[dict] = []
        for row in rows:
            meta_raw = row["meta_json"] or "{}"
            try:
                meta = json.loads(meta_raw)
            except json.JSONDecodeError:
                meta = {"raw_meta": meta_raw}
            logs.append(
                {
                    "stage": row["stage"],
                    "provider": row["provider"],
                    "latency_ms": row["latency_ms"],
                    "status": row["status"],
                    "meta": meta,
                    "created_at": row["created_at"],
                }
            )
        return logs

    def get_recent_memory_fold_telemetry(self, lookback_runs: int = 24) -> dict:
        limit = max(1, min(int(lookback_runs), 500))
        run_rows = self.conn.execute(
            """
            SELECT id
            FROM runs
            WHERE status IN ('completed', 'partial')
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        run_ids = [str(row["id"]) for row in run_rows]
        telemetry = {
            "lookback_runs": limit,
            "run_count": len(run_ids),
            "fold_count": 0,
            "degradation_count": 0,
            "degradation_rate": 0.0,
            "avg_novelty_ratio": 0.0,
            "avg_duplicate_ratio": 0.0,
        }
        if not run_ids:
            return telemetry

        placeholders = ",".join("?" for _ in run_ids)
        log_rows = self.conn.execute(
            f"""
            SELECT meta_json
            FROM stage_logs
            WHERE stage = 'memory_fold'
              AND status = 'completed'
              AND run_id IN ({placeholders})
            """,
            run_ids,
        ).fetchall()
        if not log_rows:
            return telemetry

        novelty_total = 0.0
        duplicate_total = 0.0
        fold_count = 0
        degradation_count = 0
        for row in log_rows:
            meta_raw = str(row["meta_json"] or "{}")
            try:
                meta = json.loads(meta_raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(meta, dict):
                continue
            fold_count += 1
            if bool(meta.get("degradation_detected")):
                degradation_count += 1
            novelty_total += _parse_float(meta.get("novelty_ratio"), 0.0)
            duplicate_total += _parse_float(meta.get("duplicate_ratio"), 0.0)
        if fold_count <= 0:
            return telemetry

        telemetry["fold_count"] = fold_count
        telemetry["degradation_count"] = degradation_count
        telemetry["degradation_rate"] = round(float(degradation_count) / float(fold_count), 4)
        telemetry["avg_novelty_ratio"] = round(novelty_total / float(fold_count), 4)
        telemetry["avg_duplicate_ratio"] = round(duplicate_total / float(fold_count), 4)
        return telemetry

    def get_recent_specialist_contract_telemetry(self, lookback_runs: int = 24) -> dict:
        limit = max(1, min(int(lookback_runs), 500))
        run_rows = self.conn.execute(
            """
            SELECT id, confidence_score
            FROM runs
            WHERE status IN ('completed', 'partial')
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        run_ids = [str(row["id"]) for row in run_rows]
        run_confidence_by_id = {
            str(row["id"]): _parse_float(row["confidence_score"], 0.0)
            for row in run_rows
        }
        telemetry = {
            "lookback_runs": limit,
            "run_count": len(run_ids),
            "arbitration_count": 0,
            "specialist_round_count": 0,
            "role_stats": {},
        }
        if not run_ids:
            return telemetry

        placeholders = ",".join("?" for _ in run_ids)
        log_rows = self.conn.execute(
            f"""
            SELECT run_id, meta_json
            FROM stage_logs
            WHERE stage = 'challenge_arbitration'
              AND status = 'completed'
              AND run_id IN ({placeholders})
            """,
            run_ids,
        ).fetchall()
        if not log_rows:
            return telemetry

        role_counts: dict[str, int] = {}
        role_merge_counts: dict[str, int] = {}
        role_escalate_counts: dict[str, int] = {}
        role_drop_counts: dict[str, int] = {}
        role_match_totals: dict[str, float] = {}
        role_match_counts: dict[str, int] = {}
        role_confidence_totals: dict[str, float] = {}
        role_confidence_counts: dict[str, int] = {}
        specialist_round_count = 0
        arbitration_count = 0

        for row in log_rows:
            meta_raw = str(row["meta_json"] or "{}")
            try:
                meta = json.loads(meta_raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(meta, dict):
                continue
            arbitration_count += 1
            run_id = str(row["run_id"])
            run_confidence = _parse_float(run_confidence_by_id.get(run_id), 0.0)

            role_action_counts = meta.get("specialist_role_action_counts")
            role_avg_match_scores = meta.get("specialist_role_avg_match_score")
            selected_roles = meta.get("specialist_selected_roles")
            if not isinstance(selected_roles, list) or not selected_roles:
                fallback_roles = meta.get("specialist_contract_roles")
                if isinstance(fallback_roles, list):
                    selected_roles = fallback_roles
                else:
                    selected_roles = []

            normalized_roles: list[str] = []
            for raw_role in selected_roles:
                role = str(raw_role or "").strip().lower()
                if role:
                    normalized_roles.append(role)
            specialist_round_count += len(normalized_roles)

            total_role_rounds = 0
            if isinstance(role_action_counts, dict):
                for raw_role, raw_counts in role_action_counts.items():
                    role = str(raw_role or "").strip().lower()
                    if not role or not isinstance(raw_counts, dict):
                        continue
                    merge_count = max(0, int(_parse_float(raw_counts.get("merge"), 0.0)))
                    escalate_count = max(0, int(_parse_float(raw_counts.get("escalate"), 0.0)))
                    drop_count = max(0, int(_parse_float(raw_counts.get("drop"), 0.0)))
                    total = merge_count + escalate_count + drop_count
                    if total <= 0:
                        total = max(0, int(_parse_float(raw_counts.get("total"), 0.0)))
                    if total <= 0:
                        continue
                    total_role_rounds += total
                    role_counts[role] = role_counts.get(role, 0) + total
                    role_merge_counts[role] = role_merge_counts.get(role, 0) + merge_count
                    role_escalate_counts[role] = role_escalate_counts.get(role, 0) + escalate_count
                    role_drop_counts[role] = role_drop_counts.get(role, 0) + drop_count
                    role_confidence_totals[role] = role_confidence_totals.get(role, 0.0) + (
                        run_confidence * float(total)
                    )
                    role_confidence_counts[role] = role_confidence_counts.get(role, 0) + total
            else:
                fallback_action = str(meta.get("specialist_top_action") or "merge").strip().lower()
                if fallback_action not in {"merge", "escalate", "drop"}:
                    fallback_action = "merge"
                for role in normalized_roles:
                    role_counts[role] = role_counts.get(role, 0) + 1
                    if fallback_action == "merge":
                        role_merge_counts[role] = role_merge_counts.get(role, 0) + 1
                    elif fallback_action == "escalate":
                        role_escalate_counts[role] = role_escalate_counts.get(role, 0) + 1
                    else:
                        role_drop_counts[role] = role_drop_counts.get(role, 0) + 1
                    role_confidence_totals[role] = role_confidence_totals.get(role, 0.0) + run_confidence
                    role_confidence_counts[role] = role_confidence_counts.get(role, 0) + 1

            if not normalized_roles and total_role_rounds > 0:
                specialist_round_count += total_role_rounds

            if isinstance(role_avg_match_scores, dict):
                for raw_role, raw_score in role_avg_match_scores.items():
                    role = str(raw_role or "").strip().lower()
                    if not role:
                        continue
                    role_match_totals[role] = role_match_totals.get(role, 0.0) + _parse_float(
                        raw_score,
                        0.0,
                    )
                    role_match_counts[role] = role_match_counts.get(role, 0) + 1

        role_stats: dict[str, dict[str, float | int]] = {}
        for role, selected_count in sorted(role_counts.items()):
            if selected_count <= 0:
                continue
            merge_count = role_merge_counts.get(role, 0)
            escalate_count = role_escalate_counts.get(role, 0)
            drop_count = role_drop_counts.get(role, 0)
            avg_match_score = (
                role_match_totals.get(role, 0.0) / float(role_match_counts.get(role, 0))
                if role_match_counts.get(role, 0) > 0
                else 0.0
            )
            avg_run_conf = (
                role_confidence_totals.get(role, 0.0) / float(role_confidence_counts.get(role, 0))
                if role_confidence_counts.get(role, 0) > 0
                else 0.0
            )
            role_stats[role] = {
                "selected_count": selected_count,
                "merge_count": merge_count,
                "escalate_count": escalate_count,
                "drop_count": drop_count,
                "merge_rate": round(float(merge_count) / float(selected_count), 4),
                "escalate_rate": round(float(escalate_count) / float(selected_count), 4),
                "avg_match_score": round(avg_match_score, 4),
                "avg_run_confidence": round(avg_run_conf, 4),
            }

        telemetry["arbitration_count"] = arbitration_count
        telemetry["specialist_round_count"] = specialist_round_count
        telemetry["role_stats"] = role_stats
        return telemetry
