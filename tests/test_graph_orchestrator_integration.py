# tests/test_graph_orchestrator_integration.py
from types import SimpleNamespace
import pytest

from orchestrator import DiagnosisOrchestrator
import utils.model_loader as model_loader_mod


class SimpleGraphFakeLLM:
    def __init__(self):
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        # return something generic
        return SimpleNamespace(content='{"fever":"high"}')


class FakeModelLoader:
    def __init__(self, model_provider="groq"):
        self.model_provider = model_provider

    def load_llm(self):
        return SimpleGraphFakeLLM()


@pytest.fixture(autouse=True)
def patch_model_loader(monkeypatch):
    """
    Auto-patch ModelLoader in utils.model_loader so GraphBuilder uses a fake LLM
    and does not hit external APIs during tests.
    """
    monkeypatch.setattr(model_loader_mod, "ModelLoader", FakeModelLoader)
    yield


def test_orchestrator_start_session_smoke():
    """
    Smoke test for the overall graph:
    - ensures the pipeline runs end-to-end without raising
    - does not assert strict medical correctness
    """
    orch = DiagnosisOrchestrator(model_provider="groq")
    state = orch.start_session("I have high fever and cough for 3 days.")
    # We at least expect a patient_id and some messages
    assert "patient_id" in state
    assert "messages" in state
    # If explainer was reached, diagnosis_result may be present
    # or we may be mid-loop waiting for user_response (pending_questions)
    assert ("diagnosis_result" in state) or ("pending_questions" in state)
