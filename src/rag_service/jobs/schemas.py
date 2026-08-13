"""Public, allowlisted Job representations."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

JobStatus = Literal["queued", "running", "retry_wait", "succeeded", "failed", "cancelled"]


class SafeJob(BaseModel):
    """A Job view that intentionally excludes storage and worker internals."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        hide_input_in_errors=True,
    )

    id: UUID
    operation: str = Field(min_length=1, max_length=48)
    status: JobStatus
    stage: str | None = Field(default=None, min_length=1, max_length=64)
    progress_current: int = Field(ge=0)
    progress_total: int | None = Field(default=None, ge=0)
    attempt_count: int = Field(ge=0)
    max_attempts: int = Field(ge=1, le=100)
    retryable: bool
    error_code: str | None = Field(default=None, min_length=1, max_length=64)
    error_message: str | None = Field(default=None, min_length=1, max_length=500)
    parent_job_id: UUID | None = None
    root_job_id: UUID | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None


__all__ = ["JobStatus", "SafeJob"]
