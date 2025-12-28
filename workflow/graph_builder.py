# workflow/graph_builder.py

from langgraph.graph import StateGraph, START, END
from utils.model_loader import ModelLoader
from utils.logging_config import get_logger

from agents.symptom_collector_agent import SymptomCollectorAgent
from agents.symptom_analyzer_agent import SymptomAnalyzerAgent
from agents.differential_diagnosis_agent import DifferentialDiagnosisAgent
from agents.explainer_agent import ExplainerAgent
from agents.memory_agent import MemoryAgent

log = get_logger("GraphBuilder")


class GraphBuilder:
    """
    Two-phase LangGraph pipeline (NO interrupt):

    Phase 1:
      START → memory_load → apply_user_response → collector → analyzer → diagnoser
         ├─ pending_questions → END
         └─ else → explainer → memory_persist → END

    Phase 2:
      START → memory_load → apply_user_response → analyzer → diagnoser
         ├─ pending_questions → END
         └─ else → explainer → memory_persist → END
    """

    def __init__(self, model_provider="groq", rag_vectorstore=None, llm=None):
        self.model_loader = ModelLoader(model_provider=model_provider)
        self.llm = llm or self.model_loader.load_llm()
        self.rag_vectorstore = rag_vectorstore

        self.collector = SymptomCollectorAgent(self.llm)
        self.analyzer = SymptomAnalyzerAgent()
        self.diagnoser = DifferentialDiagnosisAgent(llm=self.llm)
        self.explainer = ExplainerAgent(self.llm)
        self.memory = MemoryAgent(self.llm)

    # ---------------------------------------------------------------------

    def build_graph(self):
        graph = StateGraph(dict)

        # Nodes
        graph.add_node("memory_load", self.memory.run)
        graph.add_node("apply_user_response", self._apply_user_response_if_present)
        graph.add_node("collector", self.collector.run)
        graph.add_node("analyzer", self.analyzer.run)
        graph.add_node("diagnoser", self.diagnoser.run)
        graph.add_node("explainer", self.explainer.run)
        graph.add_node("memory_persist", self.memory.run)

        # Edges
        graph.add_edge(START, "memory_load")
        graph.add_edge("memory_load", "apply_user_response")

        graph.add_conditional_edges(
            "apply_user_response",
            self._branch_after_apply_user_response,
            {
                "initial": "collector",
                "resume": "analyzer",
            },
        )

        graph.add_edge("collector", "analyzer")
        graph.add_edge("analyzer", "diagnoser")

        graph.add_conditional_edges(
            "diagnoser",
            self._branch_after_diagnoser,
            {
                "pause": END,
                "explain": "explainer",
            },
        )

        graph.add_edge("explainer", "memory_persist")
        graph.add_edge("memory_persist", END)

        return graph.compile()

    # ---------------------------------------------------------------------

    def _branch_after_apply_user_response(self, state: dict) -> str:
        """
        Resume path if user_response exists OR symptoms already exist.
        """
        if state.get("user_response") is not None:
            return "resume"

        if state.get("symptoms"):
            return "resume"

        return "initial"

    def _branch_after_diagnoser(self, state: dict) -> str:
        pending = state.get("pending_questions") or []
        return "pause" if pending else "explain"

    # ---------------------------------------------------------------------

    def _apply_user_response_if_present(self, state: dict) -> dict:
        """
        Apply user_response → symptoms (Phase 2).
        """
        if state.get("user_response") is None:
            return state

        state = self._update_symptoms_after_answer(state)
        state["user_response"] = None  # critical: prevent reapplication
        return state

    def _update_symptoms_after_answer(self, state: dict) -> dict:
        resp = state.get("user_response")
        pending = state.get("pending_questions", [])
        symptoms = dict(state.get("symptoms", {}) or {})

        # Dict-style answers
        if isinstance(resp, dict):
            for k, v in resp.items():
                key = k
                if k.lower().startswith("do you have"):
                    key = (
                        k.replace("Do you have ", "")
                        .replace("?", "")
                        .strip()
                        .replace(" ", "_")
                        .lower()
                    )
                symptoms[key] = v

        # Free-text answer
        elif pending:
            q = pending[0]
            key = (
                q.replace("Do you have ", "")
                .replace("?", "")
                .strip()
                .replace(" ", "_")
                .lower()
            )
            val = str(resp).strip().lower()
            symptoms[key] = "yes" if val.startswith("y") else "no" if val.startswith("n") else val

        state["symptoms"] = symptoms
        state["pending_questions"] = []

        state.setdefault("messages", []).append({
            "role": "assistant",
            "content": f"[graph] applied user_response → {resp}"
        })

        log.info("Updated symptoms: %s", symptoms)
        return state

    # ---------------------------------------------------------------------

    def __call__(self):
        return self.build_graph()
