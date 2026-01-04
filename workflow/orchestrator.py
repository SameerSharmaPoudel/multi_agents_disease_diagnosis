# workflow/orchestrator.py

from typing import Optional
from dotenv import load_dotenv

from workflow.graph_builder import GraphBuilder
from utils.llm_factory import get_llm
from utils.logging_config import get_logger

log = get_logger("Orchestrator")


class DiagnosisOrchestrator:
    """
    Phase-1 Orchestrator.

    Responsibilities:
    - Load environment variables
    - Obtain LLM via get_llm()
    - Build LangGraph once
    - Expose stable entry points for FastAPI
    """

    def __init__(self, rag_vectorstore: Optional[object] = None):
        # ---------------------------------------------------------
        # Environment & LLM initialization
        # ---------------------------------------------------------
        load_dotenv()

        self.llm = get_llm()
        log.info("LLM initialized: %s", type(self.llm).__name__)

        # ---------------------------------------------------------
        # Graph construction
        # ---------------------------------------------------------
        self.builder = GraphBuilder(
            llm=self.llm,                 # overrides ModelLoader inside GraphBuilder
            rag_vectorstore=rag_vectorstore,
        )

        self.app = self.builder()
        log.info("Diagnosis graph compiled successfully")

    # ---------------------------------------------------------
    # Public API (FastAPI-facing)
    # ---------------------------------------------------------

    def start_session(
        self,
        user_initial_text: str,
        patient_id: Optional[str] = None,
    ) -> dict:
        """
        Start a new diagnosis session (Phase 1).
        """

        state = {
            "messages": [{"role": "user", "content": user_initial_text}],
            "symptoms": {},
            "pending_questions": [],
        }

        if patient_id:
            state["session_id"] = patient_id
            log.info("Starting session with session_id=%s", patient_id)

        result = self.app.invoke(
            state,
            config={"recursion_limit": 50},
        )

        # Normalize terminal status (graph controls logic)
        if result.get("status") == "running" and not result.get("pending_questions"):
            result["status"] = "completed"

        return result

    def resume_session_with_answer(
        self,
        state: dict,
        user_response,
    ) -> dict:
        """
        Resume an existing diagnosis session with user input.
        """

        state = dict(state)  # defensive copy

        state["user_response"] = user_response
        state.setdefault("messages", []).append({
            "role": "user",
            "content": str(user_response),
        })

        log.info("Resuming diagnosis session")

        result = self.app.invoke(
            state,
            config={"recursion_limit": 50},
        )

        if result.get("status") == "running" and not result.get("pending_questions"):
            result["status"] = "completed"

        return result
