from fastapi import FastAPI, HTTPException, Request
from uuid import uuid4, UUID
from fastapi.responses import JSONResponse

from api.schemas import (
    StartDiagnosisRequest,
    ContinueDiagnosisRequest,
    DiagnosisResponse,
)
from api.state_store import InMemoryStateStore
from workflow.orchestrator import DiagnosisOrchestrator
from utils.logging_config import get_logger

log = get_logger("FastAPI")

app = FastAPI(
    title="Multi-Agent Disease Diagnosis API",
    version="1.0.0",
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.exception("Unhandled exception")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"},
    )

# ------------------------------------------------------------------
# Global singletons
# ------------------------------------------------------------------

state_store = InMemoryStateStore()
orchestrator = DiagnosisOrchestrator()

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _map_state_to_response(session_id: UUID, state: dict) -> DiagnosisResponse:
    status = state.get("status", "running")
    message = "Diagnosis in progress."

    if status == "awaiting_user_input":
        pending = state.get("pending_questions") or []
        message = pending[0] if pending else "Additional information required."

    elif status == "completed":
        message = state.get("explainer_output", "Diagnosis completed.")

    return DiagnosisResponse(
        session_id=session_id,
        status=status,
        current_agent=state.get("current_agent"),
        message=message,
        data={
            "diagnosis_result": state.get("diagnosis_result"),
            "confidence": state.get("confidence"),
        } if status == "completed" else None,
    )

# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@app.post("/diagnosis/start", response_model=DiagnosisResponse)
async def start_diagnosis(request: StartDiagnosisRequest):
    """
    Start a new diagnosis session.
    """

    # 🟢 STEP 1 — prove what FastAPI ACTUALLY receives
    log.warning(
        "START REQUEST | patient_id=%r | symptoms=%r",
        request.patient_id,
        request.initial_symptoms,
    )

    # 🔴 FIX 1 — guard against UI bugs / free-text IDs
    if request.patient_id is not None and not isinstance(request.patient_id, str):
        raise HTTPException(
            status_code=400,
            detail="patient_id must be a string UUID",
        )

    # 🔴 FIX 2 — OPTIONAL but STRONGLY RECOMMENDED
    # If your UI has a "returning patient" toggle, enforce it here
    if request.patient_id is None:
        log.info("New patient visit (no patient_id provided)")
    else:
        log.info("Return visit detected for patient_id=%s", request.patient_id)

    session_id = uuid4()
    log.info("Starting new diagnosis session %s", session_id)

    try:
        state = orchestrator.start_session(
            user_initial_text=request.initial_symptoms,
            patient_id=request.patient_id,  # 🔒 structural identity only
        )
    except Exception as e:
        log.exception("Failed to start diagnosis")
        raise HTTPException(status_code=500, detail="Failed to start diagnosis") from e

    # -------------------------------------------------
    # 🔒 HARD CONTRACT WITH GRAPH
    # -------------------------------------------------
    status = state.get("status")

    if status not in ("completed", "awaiting_user_input"):
        log.error(
            "Invalid terminal state from diagnosis graph",
            extra={"status": status, "debug": state.get("debug")},
        )
        raise HTTPException(
            status_code=500,
            detail="Diagnosis engine returned an invalid state",
        )

    # Persist ONLY after graph completes its turn
    state_store.create(session_id, state)

    return _map_state_to_response(session_id, state)


@app.post("/diagnosis/continue", response_model=DiagnosisResponse)
async def continue_diagnosis(request: ContinueDiagnosisRequest):
    """
    Continue an existing diagnosis session.
    """

    session_id = request.session_id

    try:
        state = state_store.get(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    log.info(
        "Continuing session %s | patient_id=%r",
        session_id,
        state.get("patient_id"),
    )

    try:
        updated_state = orchestrator.resume_session_with_answer(
            state=state,
            user_response=request.user_answer,
        )
    except Exception as e:
        log.exception("Failed to resume diagnosis")
        raise HTTPException(status_code=500, detail="Failed to resume diagnosis") from e

    state_store.update(session_id, updated_state)
    return _map_state_to_response(session_id, updated_state)


@app.get("/health", tags=["System"])
async def health():
    return {"status": "ok"}


@app.get("/ready", tags=["System"])
async def ready():
    return {"status": "ready"}
