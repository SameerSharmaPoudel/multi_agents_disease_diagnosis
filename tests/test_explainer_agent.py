# tests/test_explainer_agent.py
import pytest
from agents.explainer_agent import ExplainerAgent


# ---------------------------------------------------------------------------
# Mock LLM class
# ---------------------------------------------------------------------------

class MockLLM:
    def __init__(self, return_text):
        self.return_text = return_text
        self.last_prompt = None

    def invoke(self, prompt):
        """Store last prompt for inspection."""
        self.last_prompt = prompt
        return type("MockResp", (), {"content": self.return_text})


class MockFailingLLM:
    def invoke(self, prompt):
        raise RuntimeError("LLM failure simulated")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_no_candidates_adds_explanation_and_session_id():
    agent = ExplainerAgent(llm=MockLLM("unused"))

    state = {
        "ranked_candidates": [],
        "symptoms": {"fever": "yes"},
        "session_id": "ABC-123",
        "messages": []
    }

    out = agent.run(state)

    assert "No diagnosis available" in out["messages"][-1]
    assert "Session Patient ID: ABC-123" in out["messages"][-1]
    assert out["session_patient_id"] == "ABC-123"


def test_llm_is_invoked_with_correct_prompt():
    mock_llm = MockLLM("diagnosis explanation")
    agent = ExplainerAgent(llm=mock_llm)

    state = {
        "ranked_candidates": [
            {"disease": "flu", "likelihood": 0.92}
        ],
        "symptoms": {"fever": "high"},
        "session_id": "PID-999",
        "messages": []
    }

    out = agent.run(state)

    # Verify LLM call
    assert mock_llm.last_prompt is not None
    assert "session ID is: PID-999" in mock_llm.last_prompt
    assert "flu" in mock_llm.last_prompt
    assert "fever" in mock_llm.last_prompt

    # Check state outputs
    assert out["messages"][-1].startswith("[explainer]")
    assert out["session_patient_id"] == "PID-999"
    assert "diagnosis_result" in out


def test_explainer_uses_fallback_on_llm_failure():
    agent = ExplainerAgent(llm=MockFailingLLM())

    state = {
        "ranked_candidates": [
            {"disease": "cold", "likelihood": 0.65}
        ],
        "symptoms": {"cough": "yes"},
        "session_id": "PID-XYZ",
        "messages": []
    }

    out = agent.run(state)

    explanation_msg = out["messages"][-1]
    assert "cold is most likely" in explanation_msg
    assert "PID-XYZ" in explanation_msg  # session ID still included


def test_session_patient_id_propagated_in_all_cases():
    agent = ExplainerAgent(llm=MockLLM("ok"))

    # Case 1 — has ranked candidates
    state1 = {
        "ranked_candidates": [{"disease": "flu", "likelihood": 0.9}],
        "symptoms": {},
        "session_id": "AAA",
        "messages": []
    }
    out1 = agent.run(state1)
    assert out1["session_patient_id"] == "AAA"

    # Case 2 — no ranked candidates
    state2 = {
        "ranked_candidates": [],
        "symptoms": {},
        "session_id": "BBB",
        "messages": []
    }
    out2 = agent.run(state2)
    assert out2["session_patient_id"] == "BBB"


def test_explainer_appends_to_messages_correctly():
    agent = ExplainerAgent(llm=MockLLM("Explanation here!"))

    state = {
        "ranked_candidates": [{"disease": "asthma", "likelihood": 0.81}],
        "symptoms": {"wheezing": "yes"},
        "session_id": "SID-777",
        "messages": [{"prev": "msg"}],
    }

    out = agent.run(state)

    assert len(out["messages"]) == 2
    last_message = out["messages"][-1]
    assert last_message.startswith("[explainer]")
    assert "Explanation here!" in last_message
