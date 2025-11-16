from utils.model_loader import ModelLoader
from prompt_library.prompt import SYSTEM_PROMPT
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.predefined import interrupt

from agents.symptom_collector_agent import SymptomCollectorAgent
from agents.symptom_analyzer_agent import SymptomAnalyzerAgent
from agents.differential_diagnosis_agent import DifferentialDiagnosisAgent
from agents.explainer_agent import ExplainerAgent
from agents.memory_agent import MemoryAgent

class GraphBuilder:
    """
    LangGraph graph that implements:
      collector -> analyzer -> diagnoser -> (maybe ask_user -> update -> analyzer) -> explainer -> memory -> END
    - ask_user is an interrupt node: the frontend must supply 'user_response' to resume.
    - update_symptoms simply takes user_response and pending_questions[0] and updates state['symptoms'].
    """

    def __init__(self, model_provider="groq", rag_vectorstore=None):
        self.model_loader = ModelLoader(model_provider=model_provider)
        self.llm = self.model_loader.load_llm()
        self.rag_vectorstore = rag_vectorstore

        # agents
        self.collector = SymptomCollectorAgent(self.llm)
        self.analyzer = SymptomAnalyzerAgent()
        self.diagnoser = DifferentialDiagnosisAgent(llm=self.llm)
        self.explainer = ExplainerAgent(self.llm)
        self.memory = MemoryAgent(self.llm)

    def build_graph(self):
        graph = StateGraph(MessagesState)

        # nodes
        graph.add_node("memory_load", self.memory.run)  # ensure patient_id & history at start
        graph.add_node("collector", self.collector.run)
        graph.add_node("analyzer", self.analyzer.run)
        graph.add_node("diagnoser", self.diagnoser.run)
        graph.add_node("ask_user", interrupt("user_response"))  # LangGraph pause here
        graph.add_node("update_symptoms", self._update_symptoms_after_answer)
        graph.add_node("explainer", self.explainer.run)
        graph.add_node("memory_persist", self.memory.run)

        # edges
        graph.add_edge(START, "memory_load")
        graph.add_edge("memory_load", "collector")
        graph.add_edge("collector", "analyzer")
        graph.add_edge("analyzer", "diagnoser")

        # branching: if diagnoser created pending_questions -> ask_user else explain
        graph.add_conditional_edges(
            "diagnoser",
            self._branch_after_diagnoser,
            {"ask": "ask_user", "explain": "explainer"}
        )

        graph.add_edge("ask_user", "update_symptoms")
        graph.add_edge("update_symptoms", "analyzer")  # loop back
        graph.add_edge("explainer", "memory_persist")
        graph.add_edge("memory_persist", END)

        return graph.compile()

    def _branch_after_diagnoser(self, state):
        """
        If diagnoser produced any pending_questions, ask user; otherwise proceed to explanation.
        Also respect the uncertainty threshold: diagnoser sets state['pending_questions'] if needed.
        """
        pending = state.get("pending_questions") or []
        if pending:
            # frontend will ask these; interrupt node will pause and wait for user_response
            return "ask"
        return "explain"

    def _update_symptoms_after_answer(self, state):
        """
        Called after ask_user interrupt. Expects state['user_response'] to contain
        either a single answer or a dict of batch answers.

        Behavior:
          - If state['pending_questions'] present and user_response is string -> map to first pending
          - If user_response is dict mapping question->answer or symptom_key->answer -> merge all
        """
        resp = state.get("user_response")
        pending = state.get("pending_questions", [])
        symptoms = state.get("symptoms", {}) or {}

        # Case 1: dict answers (from frontend batch)
        if isinstance(resp, dict):
            # Accept both natural-question->answer and symptom_key->answer mappings
            for k, v in resp.items():
                # try to convert "Do you have chest pain?" -> "chest_pain"
                key = k
                if k.lower().startswith("do you have"):
                    key = k.replace("Do you have ", "").replace("?", "").strip().replace(" ", "_")
                symptoms[key] = v
        else:
            # string answer -> assign to first pending question
            if pending:
                q = pending[0]
                key = q.replace("Do you have ", "").replace("?", "").strip().replace(" ", "_")
                # convert common answers to yes/no
                val = str(resp).strip().lower()
                if val.startswith("y"):
                    symptoms[key] = "yes"
                elif val.startswith("n"):
                    symptoms[key] = "no"
                else:
                    symptoms[key] = val

        # remove the used pending question(s) from pending_questions (simple approach)
        # frontend should supply which questions were answered in batch mode
        state["symptoms"] = symptoms
        # Clear pending questions (graph will re-run analyzer which computes new missing)
        state["pending_questions"] = []
        state.setdefault("messages", []).append(f"[update] applied user_response -> {resp}")
        return state

    def __call__(self):
        return self.build_graph()
    

# Frontend integration note: When the graph reaches ask_user interrupt, it will pause and return to your runner 
# an object indicating it is waiting for user_response. The frontend should render state['pending_questions'] 
#for the user, collect answers (single string or a dict mapping question->answer), then call the graph runner
# again with the state containing user_response
