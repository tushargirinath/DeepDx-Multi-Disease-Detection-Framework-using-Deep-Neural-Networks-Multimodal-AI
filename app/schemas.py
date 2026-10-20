"""Pydantic schemas for request/response bodies."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class PatientContext(BaseModel):
    age: Optional[str] = None
    sex: Optional[str] = None
    symptoms: Optional[str] = None
    history: Optional[str] = None
    notes: Optional[str] = None

    def non_empty(self) -> dict:
        return {k: v for k, v in self.model_dump().items() if v}


class PredictionItem(BaseModel):
    rank: int
    label: str
    probability: float
    index: int


class ReportRequest(BaseModel):
    """Everything the client needs to render a clinical PDF."""
    image_name: Optional[str] = "uploaded_image"
    demo_mode: bool = False
    predictions: list[PredictionItem] = Field(default_factory=list)
    disease_info: dict[str, Any] = Field(default_factory=dict)
    interpretation: dict[str, Any] = Field(default_factory=dict)
    patient_context: Optional[PatientContext] = None
    gradcam_overlay_b64: Optional[str] = None
