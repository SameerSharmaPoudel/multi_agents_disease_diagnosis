from utils.model_loader import ModelLoader
from prompt_library.prompt import SYSTEM_PROMPT
from langgraph.graph import StateGraph, MessagesState, END, START

from agents.symptom_collector_agent import InterviewerAgent
from agents.symptom_analyzer_agent import AnalyzerAgent
from agents.diagnosis_agent import DiagnosisAgent
from agents.lab_agent import LabAgent
from agents.explainer_agent import ExplainerAgent
from agents.memory_agent import MemoryAgent

class GraphBuilder:
    def __init__(self, model_provider: str = "groq"):
        self.model_loader = ModelLoader(model_provider=model_provider)
        self.llm = self.model_loader.load_llm()
        self.system_prompt = SYSTEM_PROMPT

        # Initialize agents
        self.interviewer_agent = InterviewerAgent(self.llm)
        self.analyzer_agent = AnalyzerAgent(self.llm)
        self.diagnosis_agent = DiagnosisAgent(self.llm)
        self.lab_agent = LabAgent(self.llm)
        self.explainer_agent = ExplainerAgent(self.llm)
        self.memory_agent = MemoryAgent(self.llm)

    def build_graph(self):
        graph = StateGraph(MessagesState)

        # Add each agent as a node in the graph
        graph.add_node("interviewer_agent", self.interviewer_agent.run)
        graph.add_node("analyzer_agent", self.analyzer_agent.run)
        graph.add_node("diagnosis_agent", self.diagnosis_agent.run)
        graph.add_node("lab_agent", self.lab_agent.run)
        graph.add_node("explainer_agent", self.explainer_agent.run)
        graph.add_node("memory_agent", self.memory_agent.run)

        # Connect the nodes sequentially
        graph.add_edge(START, "interviewer_agent")
        graph.add_edge("interviewer_agent", "analyzer_agent")
        graph.add_edge("analyzer_agent", "diagnosis_agent")
        graph.add_edge("diagnosis_agent", "lab_agent")
        graph.add_edge("lab_agent", "explainer_agent")
        graph.add_edge("explainer_agent", "memory_agent")
        graph.add_edge("memory_agent", END)

        return graph.compile()

    def __call__(self):
        return self.build_graph()