# tests/test_symptom_collector_agent.py
import pytest
from agents.symptom_collector_agent import SymptomCollectorAgent


# -------------------------------------------------------------------
# Fixtures for mocking LLM behavior
# -------------------------------------------------------------------

class MockLLM:
    """Mock that returns artificial JSON content exactly as needed."""

    def __init__(self, response):
        self.response = response

    def invoke(self, prompt):
        # Return object with .content attribute (mimic real AIMessage)
        class R:
            def __init__(self, c):
                self.content = c
        return R(self.response)


class MockLLMString:
    """Mock whose invoke() returns a raw string (no content attr)."""

    def __init__(self, response):
        self.response = response

    def invoke(self, prompt):
        return self.response


class MockLLMError:
    """Mock that raises an exception so fallback logic is triggered."""
    def invoke(self, prompt):
        raise RuntimeError("LLM failed intentionally")


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------


def test_llm_json_extraction():
    """LLM returns valid JSON -> collector must parse and use it."""
    llm = MockLLM('{"fever":"high","cough":"mild"}')
    agent = SymptomCollectorAgent(llm)

    state = {"messages": ["I have fever and cough"]}
    out = agent.run(state)

    assert out["symptoms"] == {"fever": "high", "cough": "mild"}
    assert out["messages"][-1]["agent"] == "collector"


def test_llm_string_json_extraction():
    """LLM returns raw string instead of AIMessage."""
    llm = MockLLMString('{"fever":"high"}')
    agent = SymptomCollectorAgent(llm)

    state = {"messages": ["I have fever"]}
    out = agent.run(state)

    assert out["symptoms"] == {"fever": "high"}


def test_llm_failure_triggers_fallback():
    """If LLM fails, fallback keyword extraction must run."""
    llm = MockLLMError()
    agent = SymptomCollectorAgent(llm)

    state = {"messages": ["I think I have fever and pain"]}
    out = agent.run(state)

    # fallback should detect basic symptoms
    assert out["symptoms"]["fever"] == "yes"
    assert out["symptoms"]["pain"] == "yes"


def test_merges_existing_symptoms():
    """Existing session symptoms should merge with new extraction."""
    llm = MockLLM('{"cough":"mild"}')
    agent = SymptomCollectorAgent(llm)

    state = {
        "messages": ["I have cough"],
        "symptoms": {"fever": "high"}  # existing
    }
    out = agent.run(state)

    assert out["symptoms"] == {"fever": "high", "cough": "mild"}


def test_historical_symptoms_normalization():
    """Collector should normalize historical symptoms but not activate them."""
    llm = MockLLM('{"cough":"mild"}')
    agent = SymptomCollectorAgent(llm)

    state = {
        "messages": ["I have cough"],
        "patient_history": {
            "known_symptoms": {
                "asthma": {
                    "value": "intermittent",
                    "chronic": True,
                    "first_seen": "2023-01-01T10:00:00",
                    "last_updated": "2024-01-01T10:00:00"
                }
            }
        }
    }

    out = agent.run(state)

    # current symptoms
    assert out["symptoms"]["cough"] == "mild"

    # historical symptom normalized
    assert "historical_symptoms" in out
    hist = out["historical_symptoms"]["asthma"]
    assert hist["value"] == "intermittent"
    assert hist["chronic"] is True
    assert hist["first_seen"] == "2023-01-01T10:00:00"
    assert hist["last_seen"] == "2024-01-01T10:00:00"

    # historical should NOT appear in active symptoms
    assert "asthma" not in out["symptoms"]


def test_no_messages_appended_if_no_input():
    """If there are no messages in state, collector must not break."""
    llm = MockLLM("{}")
    agent = SymptomCollectorAgent(llm)

    state = {"messages": []}
    out = agent.run(state)

    assert out["messages"] == []


def test_collector_handles_nonstring_ai_message():
    """If last message is an object with .content, collector must read content."""
    class FakeMsg:
        def __init__(self, c):
            self.content = c

    llm = MockLLM('{"fever":"high"}')
    agent = SymptomCollectorAgent(llm)

    state = {"messages": [FakeMsg("I have fever")]}
    out = agent.run(state)

    assert out["symptoms"]["fever"] == "high"


def test_collector_converts_meta_without_chronic():
    """Historical symptom without chronic key still normalizes correctly."""
    llm = MockLLM('{"fever":"high"}')
    agent = SymptomCollectorAgent(llm)

    state = {
        "messages": ["I have fever"],
        "patient_history": {
            "known_symptoms": {
                "migraine": {
                    "value": "episodic",
                    "first_seen": "2022-09-09T12:00",
                    "last_updated": "2023-03-01T08:00"
                }
            }
        }
    }

    out = agent.run(state)

    hist = out["historical_symptoms"]["migraine"]
    assert hist["value"] == "episodic"
    assert hist["chronic"] is False
    assert hist["last_seen"] == "2023-03-01T08:00"


def test_fallback_partial_detection():
    """Fallback detection picks known tokens but ignores unrelated words."""
    llm = MockLLMError()
    agent = SymptomCollectorAgent(llm)

    state = {"messages": ["My head hurts but no fever today"]}
    out = agent.run(state)

    # fallback sees "fever" and "pain" only if word appears exactly
    assert out["symptoms"]["fever"] == "yes"
    # 'pain' is not in text; 'hurts' does not match fallback
    assert "pain" not in out["symptoms"]


def test_messages_append_collector_trace():
    """Collector must always append its trace message."""
    llm = MockLLM('{"cough":"mild"}')
    agent = SymptomCollectorAgent(llm)

    state = {"messages": ["I have cough"]}
    out = agent.run(state)

    last_message = out["messages"][-1]
    assert last_message["agent"] == "collector"
    assert "extracted" in last_message["content"]
