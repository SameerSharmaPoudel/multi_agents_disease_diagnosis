from fastapi import FastAPI, HTTPException, Request
from uuid import uuid4, UUID
from fastapi.responses import JSONResponse

from app.schemas import (
    StartDiagnosisRequest,
    ContinueDiagnosisRequest,
    DiagnosisResponse,
)
from app.state_store import InMemoryStateStore
from workflow.orchestrator import DiagnosisOrchestrator
from utils.logging_config import get_logger

log = get_logger("FastAPI")

app = FastAPI(
    title="Multi-Agent Disease Diagnosis API",
    version="1.0.0",
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catch-all exception handler.
    Prevents leaking internal errors to clients.
    """
    log.exception("Unhandled exception")

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
        },
    )

# ------------------------------------------------------------------
# Global singletons (Phase-1)
# ------------------------------------------------------------------

state_store = InMemoryStateStore()
orchestrator = DiagnosisOrchestrator()

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _map_state_to_response(session_id: UUID, state: dict) -> DiagnosisResponse:
    """
    Translate internal agent state → API-safe response.
    """

    status = state.get("status", "running")

    # Default message (safe fallback)
    message = "Diagnosis in progress."

    # If waiting for user input
    if status == "awaiting_user_input":
        pending = state.get("pending_questions") or []
        message = pending[0] if pending else "Additional information required."

    # If completed, only expose explainer output
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
        }
        if status == "completed"
        else None,
    )


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------


@app.post("/diagnosis/start", response_model=DiagnosisResponse)
async def start_diagnosis(request: StartDiagnosisRequest):
    """
    Start a new diagnosis session.
    """

    session_id = uuid4()
    log.info("Starting new diagnosis session %s", session_id)

    try:
        state = orchestrator.start_session(
            user_initial_text=request.initial_symptoms,
            patient_id=request.patient_id,
        )
    except Exception as e:
        log.exception("Failed to start diagnosis")
        raise HTTPException(status_code=500, detail="Failed to start diagnosis") from e

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

    log.info("Continuing diagnosis session %s", session_id)

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
    """
    Liveness probe.
    Confirms the service process is running.
    """
    return {"status": "ok"}


@app.get("/ready", tags=["System"])
async def ready():
    """
    Readiness probe.
    Confirms the service is ready to accept traffic.
    """
    # Phase-1: always ready if process is up
    return {"status": "ready"}
