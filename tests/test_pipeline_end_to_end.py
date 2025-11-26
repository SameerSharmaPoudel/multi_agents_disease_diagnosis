import pytest
from workflow import graph_builder


# -------------------------------------------------------------------
# Fake LLM
# -------------------------------------------------------------------
class FakeLLM:
    def __init__(self):
        self.calls = []

    def invoke(self, prompt: str):
        self.calls.append(prompt)
        class Resp:
            def __init__(self, content):
                self.content = content
        return Resp("Fake LLM response")


# -------------------------------------------------------------------
# Fake Agents
# -------------------------------------------------------------------
class FakeCollectorAgent:
    def __init__(self, llm):
        self.llm = llm

    def run(self, state):
        symptoms = state.get("symptoms", {})
        symptoms.update({"fever": "high", "cough": "yes"})
        state["symptoms"] = symptoms

        state.setdefault("messages", []).append({
            "role": "assistant",
            "content": "[collector] extracted symptoms"
        })
        return state


class FakeAnalyzerAgent:
    def __init__(self, *args, **kwargs):
        pass

    def run(self, state):
        state["candidates"] = [{
            "disease": "flu",
            "jaccard": 0.9,
            "matched_symptoms": ["fever", "cough"],
            "missing_symptoms": ["headache"],
        }]
        state["missing_symptoms"] = ["headache"]

        state.setdefault("messages", []).append({
            "role": "assistant",
            "content": "[analyzer] top_candidates"
        })
        return state


class FakeDifferentialDiagnosisAgent:
    def __init__(self, llm=None, confidence_threshold=0.8, history_weight=0.15):
        self.call_count = 0

    def run(self, state):
        self.call_count += 1

        ranked = [{
            "disease": "flu",
            "likelihood": 0.95 if self.call_count > 1 else 0.5,
            "matched_symptoms": ["fever", "cough"],
            "missing_symptoms": ["headache"],
        }]
        state["ranked_candidates"] = ranked

        if self.call_count == 1:
            state["pending_questions"] = ["Do you have headache?"]
        else:
            state["pending_questions"] = []

        return state


# -------------------------------------------------------------------
# FIXED FakeExplainerAgent (robust, no KeyError)
# -------------------------------------------------------------------
class FakeExplainerAgent:
    def __init__(self, llm):
        self.llm = llm

    def run(self, state):
        ranked = state.get("ranked_candidates")

        # Safe fallback if something upstream failed
        if not ranked:
            state["diagnosis_result"] = []
            state.setdefault("messages", []).append({
                "role": "assistant",
                "content": "[explainer] No ranked candidates available."
            })
            return state

        sid = state.get("session_id", "TEST-SESSION-ID-123")
        state["diagnosis_result"] = ranked

        state.setdefault("messages", []).append({
            "role": "assistant",
            "content": f"[explainer] explanation for {sid}"
        })
        return state


# -------------------------------------------------------------------
# Fake Memory Agent
# -------------------------------------------------------------------
class FakeMemoryAgent:
    def __init__(self, llm=None, embed_model="dummy"):
        self.llm = llm

    def run(self, state):
        sid = state.get("session_id") or "TEST-SESSION-ID-123"
        state["session_id"] = sid

        if state.get("diagnosis_result"):
            state.setdefault("messages", []).append({
                "role": "assistant",
                "content": "[memory] visit persisted"
            })

        state.setdefault("patient_history", {"known_symptoms": {}, "profile": {}, "visits": []})
        return state


# -------------------------------------------------------------------
# Dummy ModelLoader
# -------------------------------------------------------------------
class DummyModelLoader:
    def __init__(self, *args, **kwargs):
        pass

    def load_llm(self):
        return FakeLLM()


# -------------------------------------------------------------------
# Fixture: patch GraphBuilder internals
# -------------------------------------------------------------------
@pytest.fixture
def orchestrator_with_fakes(monkeypatch):
    monkeypatch.setattr(graph_builder, "ModelLoader", DummyModelLoader)
    monkeypatch.setattr(graph_builder, "SymptomCollectorAgent", FakeCollectorAgent)
    monkeypatch.setattr(graph_builder, "SymptomAnalyzerAgent", FakeAnalyzerAgent)
    monkeypatch.setattr(graph_builder, "DifferentialDiagnosisAgent", FakeDifferentialDiagnosisAgent)
    monkeypatch.setattr(graph_builder, "ExplainerAgent", FakeExplainerAgent)
    monkeypatch.setattr(graph_builder, "MemoryAgent", FakeMemoryAgent)

    from workflow.orchestrator import DiagnosisOrchestrator
    return DiagnosisOrchestrator()


# -------------------------------------------------------------------
# TEST 1 — No follow-up (single-shot diagnoser)
# -------------------------------------------------------------------
def test_full_pipeline_no_followup(monkeypatch):

    monkeypatch.setattr(graph_builder, "ModelLoader", DummyModelLoader)
    monkeypatch.setattr(graph_builder, "SymptomCollectorAgent", FakeCollectorAgent)
    monkeypatch.setattr(graph_builder, "SymptomAnalyzerAgent", FakeAnalyzerAgent)
    monkeypatch.setattr(graph_builder, "ExplainerAgent", FakeExplainerAgent)
    monkeypatch.setattr(graph_builder, "MemoryAgent", FakeMemoryAgent)

    # Override diagnoser to immediately produce final result
    class SingleShot(FakeDifferentialDiagnosisAgent):
        def run(self, state):
            state["ranked_candidates"] = [{
                "disease": "flu",
                "likelihood": 0.99,
            }]
            state["pending_questions"] = []
            return state

    monkeypatch.setattr(graph_builder, "DifferentialDiagnosisAgent", SingleShot)

    gb = graph_builder.GraphBuilder()
    app = gb()

    final_state = app.invoke({"messages": [{"role": "user", "content": "fever"}]})

    assert final_state["session_id"] == "TEST-SESSION-ID-123"
    assert final_state["diagnosis_result"][0]["disease"] == "flu"
    assert any("visit persisted" in m.get("content") for m in final_state["messages"])


# -------------------------------------------------------------------
# TEST 2 — Follow-up then resume
# -------------------------------------------------------------------
def test_pipeline_followup_and_resume(orchestrator_with_fakes):

    orch = orchestrator_with_fakes

    # First call: should ask follow-up
    state1 = orch.app.invoke({"messages": [{"role": "user", "content": "fever cough"}]})

    assert state1["pending_questions"] == ["Do you have headache?"]

    # Resume session with user answer
    state1["user_response"] = "yes"
    state2 = orch.app.invoke(state1)

    assert state2["diagnosis_result"][0]["disease"] == "flu"
    assert state2["session_id"] == "TEST-SESSION-ID-123"
    assert not state2["pending_questions"]
    assert any("visit persisted" in m.get("content") for m in state2["messages"])
