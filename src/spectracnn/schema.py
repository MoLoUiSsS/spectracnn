"""Pydantic contracts for the inference API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ModulationLabel = Literal["BPSK", "QPSK", "16QAM", "64QAM"]


class IQRequest(BaseModel):
    """A single I/Q frame as ``[I, Q]`` pairs."""

    model_config = ConfigDict(extra="forbid")

    iq: list[tuple[float, float]] = Field(
        min_length=64,
        max_length=8192,
        description="Complex baseband samples as [I, Q] pairs",
        json_schema_extra={"example": [[0.12, -0.44], [0.31, 0.08]]},
    )


class PredictionResponse(BaseModel):
    modulation: ModulationLabel
    confidence: float = Field(ge=0, le=1)
    probabilities: dict[str, float]
    model_version: str
    latency_ms: float


class DemoResponse(PredictionResponse):
    true_modulation: ModulationLabel
    snr_db: float
    constellation: list[tuple[float, float]]


class ModelInfoResponse(BaseModel):
    name: str
    version: str
    trained_at: str
    classes: list[str]
    length: int
    parameters: int
    metrics: dict[str, Any]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    version: str


class RootResponse(BaseModel):
    name: str
    version: str
    docs: str
    health: str
