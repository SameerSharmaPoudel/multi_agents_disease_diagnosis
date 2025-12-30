from fastapi import APIRouter, HTTPException
from schemas import (
    StartDiagnosisRequest,
    ResumeDiagnosisRequest,
    DiagnosisResponse,
)
from workflow.run_graph import run_diagnosis_graph

router = APIRouter(prefix="/diagnosis", tags=["Diagnosis"])


def format_response(state: dict) -> DiagnosisResponse:
    pending = state.get("pending_questions", [])

    return DiagnosisResponse(
        status="awaiting_user" if pending else "completed",
        pending_questions=pending,
        explanation=state.get("final_explanation"),
        diagnosis_result=state.get("diagnosis_result"),
        session_patient_id=state.get("session_patient_id"),
        state=state,
    )


@router.post("/start", response_model=DiagnosisResponse)
def start_diagnosis(req: StartDiagnosisRequest):
    try:
        state = {
            "messages": [{"role": "user", "content": req.text}],
            "symptoms": {},
            "pending_questions": [],
            "session_id": req.patient_id,
        }

        result_state = run_diagnosis_graph(state)
        return format_response(result_state)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/resume", response_model=DiagnosisResponse)
def resume_diagnosis(req: ResumeDiagnosisRequest):
    try:
        state = dict(req.state)

        # inject user response (GraphBuilder expects this key)
        state["user_response"] = req.answer
        state.setdefault("messages", []).append({
            "role": "user",
            "content": str(req.answer),
        })

        result_state = run_diagnosis_graph(state)
        return format_response(result_state)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
