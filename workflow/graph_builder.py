# workflow/graph_builder.py

from utils.model_loader import ModelLoader
from utils.logging_config import get_logger

# ---------------------------------------------------------------
# IMPORTANT — restored imports so pytest monkeypatch works again!
# ---------------------------------------------------------------
from agents.symptom_collector_agent import SymptomCollectorAgent
from agents.symptom_analyzer_agent import SymptomAnalyzerAgent
from agents.differential_diagnosis_agent import DifferentialDiagnosisAgent
from agents.explainer_agent import ExplainerAgent
from agents.memory_agent import MemoryAgent

# LangGraph v0.8+ (NO Command, NO interrupt)
from langgraph.graph import StateGraph, MessagesState, START, END

log = get_logger("GraphBuilder")


class GraphBuilder:
    """
    LangGraph ≥0.8 pipeline (NO interrupt):
    
      memory_load → collector → analyzer → diagnoser
        → ask_user? → update_symptoms → analyzer (loop)
        → explainer → memory_persist → END
    """

    def __init__(self, model_provider="groq", rag_vectorstore=None):
        # Standard model loader
        self.model_loader = ModelLoader(model_provider=model_provider)
        self.llm = self.model_loader.load_llm()

        self.rag_vectorstore = rag_vectorstore

        # --- instantiate real agents (pytest will monkeypatch classes above)
        self.collector = SymptomCollectorAgent(self.llm)
        self.analyzer = SymptomAnalyzerAgent()
        self.diagnoser = DifferentialDiagnosisAgent(llm=self.llm)
        self.explainer = ExplainerAgent(self.llm)
        self.memory = MemoryAgent(self.llm)

    # ----------------------------------------------------------------------
    # ask_user node – NO Command, NO interrupt, just return state
    # ----------------------------------------------------------------------
    def _ask_user_node(self, state):
        """
        For langgraph<=0.8 we cannot interrupt execution,
        so we simply return state with pending questions.
        
        The orchestrator will stop execution BEFORE calling "ask_user"
        by manually checking pending_questions in diagnoser output.
        """
        log.info("ask_user node reached — returning state unchanged")
        return state

    # ----------------------------------------------------------------------
    def build_graph(self):
        graph = StateGraph(MessagesState)

        # --- nodes (pure functions)
        graph.add_node("memory_load", self.memory.run)
        graph.add_node("collector", self.collector.run)
        graph.add_node("analyzer", self.analyzer.run)
        graph.add_node("diagnoser", self.diagnoser.run)
        graph.add_node("ask_user", self._ask_user_node)
        graph.add_node("update_symptoms", self._update_symptoms_after_answer)
        graph.add_node("explainer", self.explainer.run)
        graph.add_node("memory_persist", self.memory.run)

        # --- linear start
        graph.add_edge(START, "memory_load")
        graph.add_edge("memory_load", "collector")
        graph.add_edge("collector", "analyzer")
        graph.add_edge("analyzer", "diagnoser")

        # --- branching based on missing questions
        graph.add_conditional_edges(
            "diagnoser",
            self._branch_after_diagnoser,
            {
                "ask": "ask_user",
                "explain": "explainer",
            }
        )

        # loop effects
        graph.add_edge("ask_user", "update_symptoms")
        graph.add_edge("update_symptoms", "analyzer")

        # final chain
        graph.add_edge("explainer", "memory_persist")
        graph.add_edge("memory_persist", END)

        return graph.compile()

    # ----------------------------------------------------------------------
    def _branch_after_diagnoser(self, state):
        """
        If diagnoser produced pending questions → go to ask_user,
        otherwise go to explainer.
        """
        pending = state.get("pending_questions") or []
        return "ask" if pending else "explain"

    # ----------------------------------------------------------------------
    def _update_symptoms_after_answer(self, state):
        """
        Injects user follow-up answers into symptoms.
        """
        resp = state.get("user_response")
        pending = state.get("pending_questions", [])
        symptoms = dict(state.get("symptoms", {}) or {})

        # debug info
        state.setdefault("debug", []).append({
            "agent": "graph",
            "received_user_response": resp,
            "pending_before": pending.copy(),
        })

        # direct dict response
        if isinstance(resp, dict):
            for k, v in resp.items():
                key = k
                if isinstance(k, str) and k.lower().startswith("do you have"):
                    key = (
                        k.replace("Do you have ", "")
                         .replace("?", "")
                         .strip()
                         .replace(" ", "_")
                    )
                symptoms[key] = v
        else:
            # free text -> map to first pending question
            if pending:
                q = pending[0]
                key = (
                    q.replace("Do you have ", "")
                     .replace("?", "")
                     .strip()
                     .replace(" ", "_")
                )
                val = str(resp).lower().strip()
                if val.startswith("y"):
                    symptoms[key] = "yes"
                elif val.startswith("n"):
                    symptoms[key] = "no"
                else:
                    symptoms[key] = val

        state["symptoms"] = symptoms
        state["pending_questions"] = []

        # LC-compatible assistant message
        state.setdefault("messages", []).append({
            "role": "assistant",
            "content": f"[graph] applied user_response → {resp}"
        })

        log.info(f"Updated symptoms: {symptoms}")

        return state

    # ----------------------------------------------------------------------
    def __call__(self):
        """Return compiled pipeline."""
        return self.build_graph()
