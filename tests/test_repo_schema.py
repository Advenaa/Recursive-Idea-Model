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
