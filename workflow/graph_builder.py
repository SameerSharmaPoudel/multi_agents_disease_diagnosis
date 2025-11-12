from utils.model_loader import ModelLoader
from prompt_library.prompt import SYSTEM_PROMPT
from langgraph.graph import StateGraph, MessagesState, END, START

from agents.symptom_collector_agent import SymptomCollectorAgent
from agents.symptom_analyzer_agent import AnalyzerAgent
from agents.differential_diagnosis_agent import DifferentialDiagnosisAgent
from agents.explainer_agent import ExplainerAgent
from agents.memory_agent import MemoryAgent
# from agents.lab_agent import LabAgent


class GraphBuilder:
    """
    Graph-based orchestrator that defines and manages the flow between agents.
    Incorporates a feedback loop between analyzer and diagnosis agents for iterative refinement.
    """

    def __init__(self, model_provider: str = "groq", rag_vectorstore=None):
        self.model_loader = ModelLoader(model_provider=model_provider)
        self.llm = self.model_loader.load_llm()
        self.system_prompt = SYSTEM_PROMPT
        self.rag_vectorstore = rag_vectorstore

        # Initialize agents
        self.interviewer_agent = SymptomCollectorAgent(self.llm)
        self.analyzer_agent = AnalyzerAgent(self.llm)
        self.diagnosis_agent = DifferentialDiagnosisAgent(
            rag_vectorstore=self.rag_vectorstore,
            llm=self.llm,
            max_candidates=5,
            max_rounds=3,
            confidence_threshold=0.8
        )
        # self.lab_agent = LabAgent(self.llm)
        self.explainer_agent = ExplainerAgent(self.llm)
        self.memory_agent = MemoryAgent(self.llm)

    # -------------------------------------------------------------------------
    # Graph building logic
    # -------------------------------------------------------------------------
    def build_graph(self):
        graph = StateGraph(MessagesState)

        # Add nodes (agents)
        graph.add_node("interviewer_agent", self.interviewer_agent.run)
        graph.add_node("analyzer_agent", self.analyzer_agent.run)
        graph.add_node("diagnosis_agent", self._diagnosis_with_feedback)
        graph.add_node("explainer_agent", self.explainer_agent.run)
        graph.add_node("memory_agent", self.memory_agent.run)

        # Define edges
        graph.add_edge(START, "interviewer_agent")
        graph.add_edge("interviewer_agent", "analyzer_agent")
        graph.add_edge("analyzer_agent", "diagnosis_agent")
        graph.add_edge("diagnosis_agent", "explainer_agent")
        graph.add_edge("explainer_agent", "memory_agent")
        graph.add_edge("memory_agent", END)

        return graph.compile()

    # -------------------------------------------------------------------------
    # Feedback-enhanced diagnosis integration
    # -------------------------------------------------------------------------
    def _diagnosis_with_feedback(self, state: dict):
        """
        Custom wrapper around the diagnosis agent to support iterative refinement.
        The analyzer outputs initial candidates; this function handles the feedback loop.
        """
        print("\n[GraphBuilder] Entering iterative diagnosis phase...")

        # Extract symptom and candidate info from analyzer state
        symptoms = state.get("symptoms", {})
        candidates = state.get("candidates", [])

        # Run iterative feedback loop
        result = self.diagnosis_agent.iterative_diagnosis(symptoms, candidates)

        # Update the state with refined diagnosis results
        state["diagnosis_result"] = result.get("final_candidates", [])
        state["updated_symptoms"] = result.get("final_symptoms", {})
        state["follow_up_questions"] = result.get("suggested_questions", [])

        print("[GraphBuilder] Iterative diagnosis completed.\n")
        return state

    def __call__(self):
        return self.build_graph()