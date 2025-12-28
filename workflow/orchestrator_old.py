# orchestrator.py
from workflow.graph_builder import GraphBuilder
from utils.logging_config import get_logger

log = get_logger("Orchestrator")


class DiagnosisOrchestrator:
    """
    Clean orchestrator that handles:
    - LC-safe messages
    - LangGraph Interrupt("user_response")
    - Recursive resume flow
    """

    def __init__(self, model_provider="groq", rag_vectorstore=None):
        self.builder = GraphBuilder(
            model_provider=model_provider,
            rag_vectorstore=rag_vectorstore
        )
        self.app = self.builder()

    # ------------------------------------------------------------------
    # START SESSION
    # ------------------------------------------------------------------
    def start_session(self, user_initial_text: str, patient_id: str = None) -> dict:
        """
        Prepare initial LC-compatible message and invoke the graph.
        If the graph triggers Interrupt, pytest will catch it.
        """
        init_state = {
            "messages": [
                {"role": "user", "content": user_initial_text}
            ]
        }

        if patient_id:
            init_state["patient_id"] = patient_id
            log.info(f"Starting session with provided patient_id: {patient_id}")
        else:
            log.info("Starting session (no patient_id provided)")

        # Execute graph
        state = self.app.invoke(init_state, config={"recursion_limit": 50})
        return state

    # ------------------------------------------------------------------
    # RESUME SESSION AFTER USER ANSWERS FOLLOW-UP
    # ------------------------------------------------------------------
    def resume_session_with_answer(self, state: dict, user_response) -> dict:
        """
        Takes previously interrupted state and continues the graph flow.
        """

        # Attach new user response
        state = {**state, "user_response": user_response}
        log.info(f"Resuming session with user_response={user_response}")

        # Resume graph flow
        updated = self.app.invoke(state, config={"recursion_limit": 50})
        return updated
