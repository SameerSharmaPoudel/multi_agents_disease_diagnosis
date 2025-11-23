# orchestrator.py
from graph_builder import GraphBuilder
from utils.logging_config import get_logger

log = get_logger("Orchestrator")


class DiagnosisOrchestrator:
    def __init__(self, model_provider="groq", rag_vectorstore=None):
        self.builder = GraphBuilder(model_provider=model_provider, rag_vectorstore=rag_vectorstore)
        self.app = self.builder()

    def start_session(self, user_initial_text: str, patient_id: str = None) -> dict:
        """
        Start a new session or resume if patient_id provided.
        - user_initial_text: initial free-text user message (string)
        - patient_id: optional session patient id string; if provided, memory loads history for that id
        Returns state (may be final or an interrupted state waiting for user_response).
        """
        init_state = {"messages": [user_initial_text]}
        if patient_id:
            init_state["patient_id"] = patient_id
            log.info(f"Starting session with provided patient_id: {patient_id}")
        else:
            log.info("Starting session (no patient_id provided)")

        # Invoke the compiled LangGraph app with initial state
        state = self.app.invoke(init_state)
        return state

    def resume_session_with_answer(self, state: dict, user_response) -> dict:
        """
        Resume the paused graph by injecting user_response into state and invoking again.
        - state: the paused state returned earlier by start_session or previous resume
        - user_response: either a string or dict mapping pending question -> answer
        """
        state["user_response"] = user_response
        log.info("Resuming session with user_response")
        state = self.app.invoke(state)
        return state
