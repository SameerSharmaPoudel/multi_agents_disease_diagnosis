from dotenv import load_dotenv
from workflow.graph_builder import GraphBuilder
from utils.llm_factory import get_llm

load_dotenv()

_builder = None
_app = None


def run_diagnosis_graph(state: dict) -> dict:
    global _builder, _app

    if _builder is None:
        llm = get_llm()
        _builder = GraphBuilder(llm=llm)
        _app = _builder()

    return _app.invoke(state, config={"recursion_limit": 50})