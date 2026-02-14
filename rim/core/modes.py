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


DEEP_MODE = ModeSettings(
    mode="deep",
    max_depth=4,
    critics_per_node=4,
    synthesis_passes=2,
    self_critique_pass=True,
    evidence_requirement="strict",
)

FAST_MODE = ModeSettings(
    mode="fast",
    max_depth=2,
    critics_per_node=2,
    synthesis_passes=1,
    self_critique_pass=False,
    evidence_requirement="basic",
)


def get_mode_settings(mode: ModeName) -> ModeSettings:
    if mode == "fast":
        return FAST_MODE
    return DEEP_MODE
