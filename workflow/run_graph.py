from workflow.graph_builder import GraphBuilder
from tests.fakes.fake_llm import FakeLLM

_builder = None
_app = None

def run_diagnosis_graph(state: dict) -> dict:
    global _builder, _app

    if _builder is None:
        fake_llm = FakeLLM()
        _builder = GraphBuilder(llm=fake_llm)
        _app = _builder()

    return _app.invoke(state, config={"recursion_limit": 50})