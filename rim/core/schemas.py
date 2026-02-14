from __future__ import annotations

from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    idea: str = Field(min_length=1)
    mode: Literal["deep", "fast"] = "deep"
    domain: str | None = None
    constraints: list[str] = Field(default_factory=list)
    desired_outcome: str | None = None


class DecompositionNode(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    parent_node_id: str | None = None
    depth: int = Field(ge=0)
    component_text: str = Field(min_length=1)
    node_type: Literal["claim", "assumption", "dependency"] = "claim"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class CriticFinding(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    node_id: str
    critic_type: str
    issue: str = Field(min_length=1)
    severity: Literal["low", "medium", "high", "critical"] = "medium"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    suggested_fix: str = Field(min_length=1)
    provider: Literal["codex", "claude"] | None = None


class AnalyzeResult(BaseModel):
    run_id: str
    mode: Literal["deep", "fast"]
    input_idea: str
    decomposition: list[DecompositionNode]
    critic_findings: list[CriticFinding]
    synthesized_idea: str
    changes_summary: list[str]
    residual_risks: list[str]
    next_experiments: list[str]
    confidence_score: float = Field(ge=0.0, le=1.0)


class RunError(BaseModel):
    stage: str
    provider: str | None = None
    message: str
    retryable: bool = False


class AnalyzeRunResponse(BaseModel):
    run_id: str
    status: Literal["queued", "running", "completed", "failed", "partial"]
    result: AnalyzeResult | None = None
    error_summary: str | None = None
    error: RunError | None = None


class RunSummary(BaseModel):
    run_id: str
    mode: Literal["deep", "fast"]
    input_idea: str
    status: Literal["queued", "running", "completed", "failed", "partial"]
    created_at: str
    completed_at: str | None = None
    confidence_score: float | None = None
    error_summary: str | None = None


class RunListResponse(BaseModel):
    runs: list[RunSummary]
    count: int
    limit: int
    offset: int
    status: str | None = None
    mode: str | None = None


class StageLogEntry(BaseModel):
    stage: str
    provider: str | None = None
    latency_ms: int | None = None
    status: str
    meta: dict = Field(default_factory=dict)
    created_at: str


class RunLogsResponse(BaseModel):
    run_id: str
    logs: list[StageLogEntry]


class RunFeedbackRequest(BaseModel):
    verdict: Literal["accept", "reject"]
    notes: str | None = None


class RunFeedbackResponse(BaseModel):
    run_id: str
    verdict: Literal["accept", "reject"]
    notes: str | None = None
    updated_memory_entries: int
    created_at: str


class HealthResponse(BaseModel):
    ok: bool
    db: bool
    providers: dict[str, bool]
