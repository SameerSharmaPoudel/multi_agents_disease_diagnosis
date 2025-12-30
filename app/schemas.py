from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class StartDiagnosisRequest(BaseModel):
    text: str = Field(..., description="Initial symptom description")
    patient_id: Optional[str] = Field(None, description="Optional patient/session ID")


class ResumeDiagnosisRequest(BaseModel):
    state: Dict[str, Any] = Field(..., description="State returned by previous API call")
    answer: Any = Field(..., description="User answer to pending question(s)")


class DiagnosisResponse(BaseModel):
    status: str
    pending_questions: List[str]
    explanation: Optional[str]
    diagnosis_result: Optional[Any]
    session_patient_id: Optional[str]
    state: Dict[str, Any]