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
    Option A GraphBuilder
    """

    def __init__(self, llm, rag_vectorstore=None):
        if llm is None:
            raise ValueError("GraphBuilder requires an initialized LLM")

        self.llm = llm

        self.memory = MemoryAgent(self.llm)
        self.collector = SymptomCollectorAgent(self.llm)
        self.analyzer = SymptomAnalyzerAgent()
        self.diagnoser = DifferentialDiagnosisAgent(llm=self.llm)
        self.explainer = ExplainerAgent(self.llm)

    def build_graph(self):
        graph = StateGraph(dict)

        graph.add_node("memory_load", self.memory.run)
        graph.add_node("collector", self.collector.run)
        graph.add_node("analyzer", self.analyzer.run)
        graph.add_node("diagnoser", self.diagnoser.run)
        graph.add_node("explainer", self._finalize_and_explain)
        graph.add_node("memory_persist", self.memory.run)
        graph.add_node("ensure_terminal_status", self._ensure_terminal_status)

        graph.add_edge(START, "memory_load")
        graph.add_edge("memory_load", "collector")
        graph.add_edge("collector", "analyzer")
        graph.add_edge("analyzer", "diagnoser")

        graph.add_conditional_edges(
            "diagnoser",
            self._route_after_diagnosis,
            {
                "pause": "ensure_terminal_status",
                "explain": "explainer",
            },
        )

        graph.add_edge("explainer", "memory_persist")
        graph.add_edge("memory_persist", "ensure_terminal_status")
        graph.add_edge("ensure_terminal_status", END)

        return graph.compile()

    def _route_after_diagnosis(self, state: dict) -> str:
        if state.get("pending_questions"):
            return "pause"
        return "explain"

    # ---------------------------------------------------------
    # TERMINAL GUARD (FIXED)
    # ---------------------------------------------------------
    def _ensure_terminal_status(self, state: dict) -> dict:
        symptoms = state.get("symptoms") or []
        pending = state.get("pending_questions") or []
        final = state.get("diagnosis_result") or []

        # 🔒 Finalization invariant
        if state.get("diagnosis_finalized"):
            state["status"] = "completed"
            log.info("Terminal status enforced: completed (finalized)")
            return state

        if not symptoms:
            state["pending_questions"] = ["Please describe your symptoms."]
            state["status"] = "awaiting_user_input"
            log.warning("Blocked completion: no symptoms")
            return state

        if pending:
            state["status"] = "awaiting_user_input"
            log.info("Awaiting user input for pending questions")
            return state

        if not final:
            state["pending_questions"] = [
                "I need a bit more information to assess your condition."
            ]
            state["status"] = "awaiting_user_input"
            log.warning("Blocked completion: no diagnosis result")
            return state

        state["status"] = "completed"
        log.info("Terminal status enforced: completed")
        return state

    def _finalize_and_explain(self, state: dict) -> dict:
        return self.explainer.run(state)

    def __call__(self):
        return self.build_graph()
