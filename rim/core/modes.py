from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ModeName = Literal["deep", "fast"]


@dataclass(frozen=True)
class ModeSettings:
    mode: ModeName
    max_depth: int
    critics_per_node: int
    synthesis_passes: int
    self_critique_pass: bool
    evidence_requirement: str
    confidence_stop_threshold: float
    marginal_gain_min_new_nodes: int
    marginal_gain_patience: int
    max_children_per_node: int
    runtime_budget_sec: int


DEEP_MODE = ModeSettings(
    mode="deep",
    max_depth=4,
    critics_per_node=4,
    synthesis_passes=2,
    self_critique_pass=True,
    evidence_requirement="strict",
    confidence_stop_threshold=0.85,
    marginal_gain_min_new_nodes=2,
    marginal_gain_patience=2,
    max_children_per_node=4,
    runtime_budget_sec=420,
)

FAST_MODE = ModeSettings(
    mode="fast",
    max_depth=2,
    critics_per_node=2,
    synthesis_passes=1,
    self_critique_pass=False,
    evidence_requirement="basic",
    confidence_stop_threshold=0.92,
    marginal_gain_min_new_nodes=1,
    marginal_gain_patience=1,
    max_children_per_node=2,
    runtime_budget_sec=120,
)


def get_mode_settings(mode: ModeName) -> ModeSettings:
    if mode == "fast":
        return FAST_MODE
    return DEEP_MODE
