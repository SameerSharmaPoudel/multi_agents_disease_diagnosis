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
    Invariant-safe LangGraph pipeline.

    HARD GUARANTEES:
    - status is ALWAYS one of:
        - awaiting_user_input
        - completed
    - Graph NEVER terminates without status
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

        # 🔒 TERMINAL GUARD (NEW)
        graph.add_node("ensure_terminal_status", self._ensure_terminal_status)

        # =====================================================
        # START branching
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

        # -----------------
        # Branch after diagnoser
        # -----------------
        graph.add_conditional_edges(
            "diagnoser",
            self._branch_after_diagnoser,
            {
                "pause": "ensure_terminal_status",
                "explain": "explainer",
            },
        )

        graph.add_edge("explainer", "memory_persist")
        graph.add_edge("memory_persist", "ensure_terminal_status")

        # 🔚 ONLY exit point
        graph.add_edge("ensure_terminal_status", END)

        return graph.compile()

    # ---------------------------------------------------------
    # Branching (NO invariants here)
    # ---------------------------------------------------------

    def _branch_after_diagnoser(self, state: dict) -> str:
        """
        Routing only.
        NO guarantees here.
        """

        if state.get("pending_questions"):
            return "pause"

        return "explain"

    # ---------------------------------------------------------
    # 🔒 TERMINAL GUARD (THE FIX)
    # ---------------------------------------------------------

    def _ensure_terminal_status(self, state: dict) -> dict:
        """
        Enforces terminal invariants.
        ALWAYS runs before END.
        """

        if state.get("status") is None:
            if state.get("pending_questions"):
                state["status"] = "awaiting_user_input"
            else:
                state["status"] = "completed"

        # Optional safety logging
        log.info("Terminal status enforced: %s", state["status"])
        return state

    # ---------------------------------------------------------
    # Finalization
    # ---------------------------------------------------------

    def _finalize_and_explain(self, state: dict) -> dict:
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

        if not pending:
            return state

        q = pending[0]
        key = (
            q.replace("Do you have ", "")
            .replace("?", "")
            .strip()
            .replace(" ", "_")
            .lower()
        )

        val = str(resp).strip().lower()

        if val.startswith("y"):
            symptoms[key] = "yes"
        elif val.startswith("n"):
            symptoms[key] = "no"
        else:
            symptoms[key] = val

        state["symptoms"] = symptoms
        state["pending_questions"] = []
        state["user_response"] = None

        log.info("Updated symptoms: %s", symptoms)
        return state

    def __call__(self):
        return self.build_graph()
