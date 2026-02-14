from datetime import datetime, timedelta, timezone
from pathlib import Path

from rim.storage.repo import RunRepository


def test_repository_initializes_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "rim_test.db"
    repo = RunRepository(db_path=db_path)
    assert repo.healthcheck()

    tables = {
        row[0]
        for row in repo.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert "runs" in tables
    assert "nodes" in tables
    assert "critic_findings" in tables
    assert "run_feedback" in tables


def test_memory_context_ordering(tmp_path: Path) -> None:
    db_path = tmp_path / "rim_test.db"
    repo = RunRepository(db_path=db_path)
    repo.save_memory_entries(
        run_id="run-1",
        entries=[
            {"entry_type": "insight", "entry_text": "low", "score": 0.2},
            {"entry_type": "insight", "entry_text": "high", "score": 0.9},
            {"entry_type": "insight", "entry_text": "mid", "score": 0.5},
        ],
    )

    top = repo.get_memory_context(limit=2)
    assert top == ["high", "mid"]


def test_memory_context_filters_domain_recency_severity(tmp_path: Path) -> None:
    db_path = tmp_path / "rim_test.db"
    repo = RunRepository(db_path=db_path)
    repo.save_memory_entries(
        run_id="run-1",
        entries=[
            {
                "entry_type": "pattern",
                "entry_text": "fin-high",
                "domain": "finance",
                "severity": "high",
                "score": 0.8,
            },
            {
                "entry_type": "insight",
                "entry_text": "fin-low",
                "domain": "finance",
                "severity": "low",
                "score": 0.9,
            },
            {
                "entry_type": "pattern",
                "entry_text": "global-high",
                "severity": "high",
                "score": 0.7,
            },
            {
                "entry_type": "pattern",
                "entry_text": "prod-critical",
                "domain": "product",
                "severity": "critical",
                "score": 0.95,
            },
            {
                "entry_type": "pattern",
                "entry_text": "fin-old-critical",
                "domain": "finance",
                "severity": "critical",
                "score": 0.99,
            },
        ],
    )
    old_date = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
    repo.conn.execute(
        "UPDATE memory_entries SET created_at = ? WHERE entry_text = 'fin-old-critical'",
        (old_date,),
    )
    repo.conn.commit()

    filtered = repo.get_memory_context(
        limit=10,
        domain="finance",
        max_age_days=30,
        min_severity="high",
    )
    assert filtered == ["fin-high", "global-high"]


def test_stage_logs_roundtrip(tmp_path: Path) -> None:
    db_path = tmp_path / "rim_test.db"
    repo = RunRepository(db_path=db_path)
    repo.create_run("run-2", "deep", "idea")
    repo.log_stage(
        run_id="run-2",
        stage="decompose",
        status="completed",
        provider="codex",
        latency_ms=123,
        meta={"node_count": 4},
    )
    logs = repo.get_stage_logs("run-2")
    assert len(logs) == 1
    assert logs[0]["stage"] == "decompose"
    assert logs[0]["provider"] == "codex"
    assert logs[0]["latency_ms"] == 123
    assert logs[0]["meta"]["node_count"] == 4


def test_pending_runs_and_request_roundtrip(tmp_path: Path) -> None:
    db_path = tmp_path / "rim_test.db"
    repo = RunRepository(db_path=db_path)
    repo.create_run_with_request(
        run_id="run-queued",
        mode="deep",
        input_idea="idea 1",
        request_json='{"idea":"idea 1","mode":"deep"}',
        status="queued",
    )
    repo.create_run_with_request(
        run_id="run-running",
        mode="fast",
        input_idea="idea 2",
        request_json='{"idea":"idea 2","mode":"fast"}',
        status="running",
    )
    repo.create_run_with_request(
        run_id="run-done",
        mode="fast",
        input_idea="idea 3",
        request_json='{"idea":"idea 3","mode":"fast"}',
        status="completed",
    )

    pending = repo.get_pending_runs()
    assert pending == ["run-queued", "run-running"]
    req = repo.get_run_request("run-queued")
    assert req is not None
    assert req["idea"] == "idea 1"


def test_list_runs_filters_and_pagination(tmp_path: Path) -> None:
    db_path = tmp_path / "rim_test.db"
    repo = RunRepository(db_path=db_path)
    repo.create_run_with_request(
        run_id="run-a",
        mode="deep",
        input_idea="idea a",
        request_json='{"idea":"idea a","mode":"deep"}',
        status="completed",
    )
    repo.create_run_with_request(
        run_id="run-b",
        mode="fast",
        input_idea="idea b",
        request_json='{"idea":"idea b","mode":"fast"}',
        status="failed",
    )
    repo.create_run_with_request(
        run_id="run-c",
        mode="deep",
        input_idea="idea c",
        request_json='{"idea":"idea c","mode":"deep"}',
        status="queued",
    )

    deep_only = repo.list_runs(mode="deep", limit=10, offset=0)
    assert [row["id"] for row in deep_only] == ["run-c", "run-a"]

    failed_only = repo.list_runs(status="failed", limit=10, offset=0)
    assert [row["id"] for row in failed_only] == ["run-b"]

    page_one = repo.list_runs(limit=2, offset=0)
    page_two = repo.list_runs(limit=2, offset=2)
    assert [row["id"] for row in page_one] == ["run-c", "run-b"]
    assert [row["id"] for row in page_two] == ["run-a"]


def test_feedback_updates_memory_scores_and_creates_feedback_entry(tmp_path: Path) -> None:
    db_path = tmp_path / "rim_test.db"
    repo = RunRepository(db_path=db_path)
    repo.create_run_with_request(
        run_id="run-feedback",
        mode="deep",
        input_idea="idea",
        request_json='{"idea":"idea","mode":"deep","domain":"finance"}',
        status="completed",
    )
    repo.save_memory_entries(
        run_id="run-feedback",
        entries=[
            {
                "entry_type": "insight",
                "entry_text": "insight-a",
                "domain": "finance",
                "severity": "medium",
                "score": 0.5,
            },
            {
                "entry_type": "failure",
                "entry_text": "risk-a",
                "domain": "finance",
                "severity": "high",
                "score": 0.4,
            },
        ],
    )

    result = repo.submit_run_feedback(
        run_id="run-feedback",
        verdict="accept",
        notes="useful output",
    )
    assert result["run_id"] == "run-feedback"
    assert result["verdict"] == "accept"
    assert result["updated_memory_entries"] == 3

    entries = repo.conn.execute(
        "SELECT entry_type, entry_text, score, domain, severity FROM memory_entries WHERE run_id = ?",
        ("run-feedback",),
    ).fetchall()
    by_text = {row["entry_text"]: row for row in entries}
    assert float(by_text["insight-a"]["score"]) > 0.5
    assert float(by_text["risk-a"]["score"]) > 0.4
    assert "User feedback (accept): useful output" in by_text
    assert by_text["User feedback (accept): useful output"]["domain"] == "finance"
