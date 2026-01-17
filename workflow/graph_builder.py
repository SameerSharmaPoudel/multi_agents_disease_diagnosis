# workflow/graph_builder.py

from langgraph.graph import StateGraph, START, END
from utils.logging_config import get_logger

from agents.symptom_collector_agent import SymptomCollectorAgent
from agents.symptom_analyzer_agent import SymptomAnalyzerAgent
from agents.differential_diagnosis_agent import DifferentialDiagnosisAgent
from agents.explainer_agent import ExplainerAgent
from agents.memory_agent import MemoryAgent

log = get_logger("GraphBuilder")


class GraphBuilder:
    """
    Correct, invariant-safe LangGraph pipeline.

    HARD GUARANTEES:
    - SymptomCollector ALWAYS runs on first visit
    - Analyzer ALWAYS runs after symptoms exist
    - Diagnoser NEVER finalizes without symptoms
    - status is ALWAYS one of:
        - awaiting_user_input
        - completed
    """

    def __init__(self, llm, rag_vectorstore=None):
        if llm is None:
            raise ValueError("GraphBuilder requires an initialized LLM")

        self.llm = llm
        self.rag_vectorstore = rag_vectorstore

        # Agents
        self.collector = SymptomCollectorAgent(self.llm)
        self.analyzer = SymptomAnalyzerAgent()
        self.diagnoser = DifferentialDiagnosisAgent(llm=self.llm)
        self.explainer = ExplainerAgent(self.llm)
        self.memory = MemoryAgent(self.llm)

    # ---------------------------------------------------------
    # Graph construction
    # ---------------------------------------------------------

    def build_graph(self):
        graph = StateGraph(dict)

        # -----------------
        # Nodes
        # -----------------
        graph.add_node("memory_load", self.memory.run)
        graph.add_node("apply_user_response", self._apply_user_response_if_present)
        graph.add_node("collector", self.collector.run)
        graph.add_node("analyzer", self.analyzer.run)
        graph.add_node("diagnoser", self.diagnoser.run)
        graph.add_node("explainer", self._finalize_and_explain)
        graph.add_node("memory_persist", self.memory.run)

        # =====================================================
        # 🔑 CRITICAL: branch immediately at START
        # =====================================================
        graph.add_conditional_edges(
            START,
            lambda state: "resume" if state.get("user_response") else "start",
            {
                "start": "memory_load",
                "resume": "apply_user_response",
            },
        )

        # -----------------
        # New session path
        # -----------------
        graph.add_edge("memory_load", "collector")
        graph.add_edge("collector", "analyzer")

        # -----------------
        # Resume path
        # -----------------
        graph.add_edge("apply_user_response", "analyzer")

        # -----------------
        # Shared path
        # -----------------
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

    # ---------------------------------------------------------
    # 🔒 HARD INVARIANT ENFORCEMENT
    # ---------------------------------------------------------

    def _branch_after_diagnoser(self, state: dict) -> str:
        """
        Decide whether to pause or explain.
        GUARANTEES terminal status.
        """

        symptoms = state.get("symptoms") or {}
        pending = state.get("pending_questions")

        # 🔴 Case 1: No symptoms extracted at all
        if not symptoms:
            state["pending_questions"] = ["Please describe your symptoms."]
            state["status"] = "awaiting_user_input"
            return "pause"

        # 🟡 Case 2: Diagnoser explicitly requests more info
        if pending:
            state["status"] = "awaiting_user_input"
            return "pause"

        # 🟢 Case 3: Safe to explain
        return "explain"

    # ---------------------------------------------------------
    # Finalization
    # ---------------------------------------------------------

    def _finalize_and_explain(self, state: dict) -> dict:
        """
        Explainer + terminal normalization.
        """
        state = self.explainer.run(state)
        state["status"] = "completed"
        return state

    # ---------------------------------------------------------
    # Resume handling
    # ---------------------------------------------------------

    def _apply_user_response_if_present(self, state: dict) -> dict:
        if state.get("user_response") is None:
            return state

        state = self._update_symptoms_after_answer(state)
        state["user_response"] = None
        return state

    def _update_symptoms_after_answer(self, state: dict) -> dict:
        resp = state.get("user_response")
        pending = state.get("pending_questions", [])
        symptoms = dict(state.get("symptoms", {}) or {})

        if isinstance(resp, dict):
            symptoms.update(resp)

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
            symptoms[key] = (
                "yes" if val.startswith("y")
                else "no" if val.startswith("n")
                else val
            )

        state["symptoms"] = symptoms
        state["pending_questions"] = []

        log.info("Updated symptoms: %s", symptoms)
        return state

    def __call__(self):
        return self.build_graph()
