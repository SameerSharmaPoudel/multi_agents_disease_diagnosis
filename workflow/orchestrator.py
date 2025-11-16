from graph_builder import GraphBuilder

class DiagnosisOrchestrator:
    def __init__(self, model_provider="groq", rag_vectorstore=None):
        self.builder = GraphBuilder(model_provider=model_provider, rag_vectorstore=rag_vectorstore)
        self.app = self.builder()

    def start_session(self, user_initial_text: str) -> dict:
        """
        Start a new session: pass initial user message text.
        Returns state; if waiting for user answers, frontend must collect answers and call resume_session().
        """
        init_state = {"messages": [user_initial_text]}
        state = self.app.invoke(init_state)
        # Typically state will either be final or waiting for user_response via interrupt
        return state

    def resume_session_with_answer(self, state: dict, user_response):
        """
        Resume the paused graph by injecting user_response into state and invoking again.
        user_response: either a string or dict mapping questions->answers
        """
        state["user_response"] = user_response
        # Reinvoke graph; it will proceed from the interrupt node
        state = self.app.invoke(state)
        return state
    
#For real frontends, maintain state server-side keyed by session_id (the patient_id produced by MemoryAgent).
# When ask_user is reached, send state['pending_questions'] to the client, collect answers, 
# call resume_session_with_answer(state, answers_dict_or_string)