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
