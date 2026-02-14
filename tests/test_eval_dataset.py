from __future__ import annotations

from rim.eval.runner import DEFAULT_DATASET_PATH, load_dataset


def test_default_benchmark_dataset_has_canonical_size_and_unique_ids() -> None:
    dataset = load_dataset(DEFAULT_DATASET_PATH)
    assert len(dataset) == 20
    ids = [str(item.get("id")) for item in dataset]
    assert len(ids) == len(set(ids))
    assert all(str(item.get("idea") or "").strip() for item in dataset)
