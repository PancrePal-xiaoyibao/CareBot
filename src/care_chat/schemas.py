from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field


CareRoleHint = Literal["auto", "patient", "caregiver", "volunteer"]


@dataclass(slots=True)
class CareChatContext:
    role_hint: CareRoleHint = "auto"
    session_id: str | None = None


class InputSafetyAssessment(BaseModel):
    tripwire_triggered: bool = Field(
        description="Whether immediate offline crisis escalation is needed."
    )
    category: Literal["none", "medical_emergency", "self_harm"] = Field(
        description="The safety bucket for the user input."
    )
    reason: str = Field(description="Short explanation for the decision.")


class OutputSafetyAssessment(BaseModel):
    unsafe: bool = Field(description="Whether the assistant response is medically unsafe.")
    reason: str = Field(description="Short explanation for the decision.")


class CrisisAssessment(BaseModel):
    risk_level: Literal["none", "low", "moderate", "high", "critical"] = Field(
        description="用户自伤风险等级"
    )
    reason: str = Field(description="风险评估的简要依据")
