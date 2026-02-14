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
