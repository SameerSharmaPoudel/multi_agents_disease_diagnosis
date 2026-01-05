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

    session_id: UUID = Field(
        ...,
        example="f47ac10b-58cc-4372-a567-0e02b2c3d479"
    )

    status: Literal[
        "started",
        "running",
        "awaiting_user_input",
        "completed",
        "failed",
    ] = Field(
        ...,
        example="awaiting_user_input"
    )

    current_agent: Optional[str] = Field(
        None,
        description="Name of the agent currently in control",
        example="SymptomAnalyzerAgent"
    )

    message: str = Field(
        ...,
        description="User-facing message (always safe to display)",
        example="Do you have chest pain or shortness of breath?"
    )

    data: Optional[Dict[str, Any]] = Field(
        None,
        description="Structured payload (diagnosis, confidence, etc.)",
        example={
            "diagnosis_result": "Migraine",
            "confidence": 0.82
        }
    )
