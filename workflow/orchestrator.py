# workflow/orchestrator.py

from typing import Optional
from dotenv import load_dotenv

from workflow.graph_builder import GraphBuilder
from utils.llm_factory import get_llm
from utils.logging_config import get_logger

log = get_logger("Orchestrator")


class DiagnosisOrchestrator:
    """
    Phase-1 Orchestrator (consistent).

    Owns:
    - LLM initialization
    - Graph lifecycle
    """

    def __init__(self, rag_vectorstore: Optional[object] = None):
        load_dotenv()

        self.llm = get_llm()
        log.info("LLM initialized: %s", type(self.llm).__name__)

        self.builder = GraphBuilder(
            llm=self.llm,
            rag_vectorstore=rag_vectorstore,
        )

        self.app = self.builder()
        log.info("Diagnosis graph compiled successfully")

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def start_session(
        self,
        user_initial_text: str,
        patient_id: Optional[str] = None,
    ) -> dict:
        """
        Start a new diagnosis workflow.
        """

        state = {
            "messages": [{"role": "user", "content": user_initial_text}],
            "symptoms": {},
            "pending_questions": [],
        }

        if patient_id:
            state["patient_id"] = patient_id
            log.info("Starting diagnosis for existing patient_id=%s", patient_id)
        else:
            log.info("Starting diagnosis for new patient")

        result = self.app.invoke(
            state,
            config={"recursion_limit": 50},
        )

        return result

    def resume_session_with_answer(
        self,
        state: dict,
        user_response,
    ) -> dict:
        """
        Resume diagnosis with user input.
        """

        state = dict(state)

        state["user_response"] = user_response
        state.setdefault("messages", []).append({
            "role": "user",
            "content": str(user_response),
        })

        log.info("Resuming diagnosis workflow")

        result = self.app.invoke(
            state,
            config={"recursion_limit": 50},
        )

        return result
