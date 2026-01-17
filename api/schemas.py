from pydantic import BaseModel, Field
from typing import Optional, Literal, Dict, Any
from uuid import UUID

# -------------------------
# Requests
# -------------------------

class StartDiagnosisRequest(BaseModel):
    """
    Request to start a new diagnosis session.
    """

    initial_symptoms: str = Field(
        ...,
        description="Free-text description of initial symptoms",
        example="I have had a fever and headache for the last two days"
    )

    patient_id: Optional[str] = Field(
        None,
        description="Optional persistent patient identifier for revisits",
        example="93cf3f3e-9d1d-4554-8023-886cdcfe516e"
    )


class ContinueDiagnosisRequest(BaseModel):
    """
    Request to continue an existing diagnosis session.
    """

    session_id: UUID = Field(
        ...,
        description="Diagnosis session identifier",
        example="f47ac10b-58cc-4372-a567-0e02b2c3d479"
    )

    user_answer: str = Field(
        ...,
        description="User response to the agent's question",
        example="Yes, I feel pain when breathing deeply"
    )

# -------------------------
# Responses
# -------------------------

class DiagnosisResponse(BaseModel):
    """
    Unified response returned by all diagnosis endpoints.
    """

    session_id: UUID

    status: Literal[
        "running",
        "awaiting_user_input",
        "completed",
        "failed",
    ]

    current_agent: Optional[str] = Field(
        None,
        description="Name of the agent currently in control",
    )

    message: str = Field(
        ...,
        description="User-facing message (always safe to display)",
    )

    data: Optional[Dict[str, Any]] = Field(
        None,
        description="Structured payload (diagnosis, confidence, etc.)",
    )
