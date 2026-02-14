from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from rim.core.schemas import CriticFinding, DecompositionNode
from rim.storage.db import get_connection, init_db


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        self.conn.execute(
            """
            INSERT INTO runs (id, mode, input_idea, status, created_at)
            VALUES (?, ?, ?, 'running', ?)
            """,
            (run_id, mode, input_idea, _utc_now()),
        )
        self.conn.commit()

    def mark_run_status(
        self,
        run_id: str,
        status: str,
        confidence_score: float | None = None,
        error_summary: str | None = None,
    ) -> None:
        self.conn.execute(
            """
            UPDATE runs
            SET status = ?, completed_at = ?, confidence_score = ?, error_summary = ?
            WHERE id = ?
            """,
            (status, _utc_now(), confidence_score, error_summary, run_id),
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

        return {
            "run_id": run_row["id"],
            "status": run_row["status"],
            "error_summary": run_row["error_summary"],
            "result": result_payload,
        }
