# workflow/orchestrator.py

from workflow.graph_builder import GraphBuilder
from utils.logging_config import get_logger

log = get_logger("Orchestrator")


class DiagnosisOrchestrator:
    """
    Two-phase orchestrator (NO interrupt).
    """

    def __init__(self, model_provider="groq", rag_vectorstore=None):
        self.builder = GraphBuilder(model_provider=model_provider, rag_vectorstore=rag_vectorstore)
        self.app = self.builder()

    def start_session(self, user_initial_text: str, patient_id: str | None = None) -> dict:
        state = {
            "messages": [{"role": "user", "content": user_initial_text}],
            "symptoms": {},
            "pending_questions": [],
        }

        if patient_id:
            state["patient_id"] = patient_id
            log.info("Starting session with patient_id=%s", patient_id)

        return self.app.invoke(state, config={"recursion_limit": 50})

    def resume_session_with_answer(self, state: dict, user_response) -> dict:
        """
        Resume Phase 2 by injecting user_response.
        """
        state = dict(state)

        state["user_response"] = user_response
        state.setdefault("messages", []).append({
            "role": "user",
            "content": str(user_response),
        })

        log.info("Resuming with user_response=%s", user_response)

        return self.app.invoke(state, config={"recursion_limit": 50})
