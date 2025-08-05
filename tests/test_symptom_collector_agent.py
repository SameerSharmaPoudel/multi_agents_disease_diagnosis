import pytest
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import ValidationError
from typing import List

from agents.symptom_collector_agent import SymptomCollectorAgent, SymptomInfo, REQUIRED_SYMPTOMS

# Mock LLM that returns responses in sequence
class FakeListLLM:
    def __init__(self, responses: List[str]):
        self.responses = responses
        self.call_count = 0

    def invoke(self, _):
        # Return next response in sequence
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
            return AIMessage(content=resp)
        else:
            raise RuntimeError("No more mock responses available.")

# === TEST CASE ===
def test_agent_multi_turn_completion():
    
    # Step 1: Fake LLM simulates:
    #   First run → incomplete JSON
    #   Second run → complete JSON
    fake_llm = FakeListLLM(responses=[
        '{"fatigue": "yes", "pain": "chest"}',  # Missing many fields → incomplete
        '{"fever": "high", "cough": "present", "fatigue": "yes", "pain": "chest", "duration": "2 days", "location": "chest"}'
    ])

    agent = SymptomCollectorAgent(llm=fake_llm)

    # Step 2: First turn (incomplete input)
    state = {"messages": [HumanMessage(content="I feel tired and my chest hurts")]}
    result1 = agent.run(state)

    # Should be incomplete
    assert result1["agent_status"] == "incomplete"
    assert result1["symptoms"] == {}  # No valid structured output
    assert isinstance(result1["messages"][-1], AIMessage)  # Clarification question present

    # Step 3: Second turn (user replies with missing info)
    clarification_reply = HumanMessage(content="I also have a high fever, cough, for 2 days in my chest area")
    state2 = {"messages": result1["messages"] + [clarification_reply]}
    result2 = agent.run(state2)

    # Should now be complete
    assert result2["agent_status"] == "complete"
    for sym in REQUIRED_SYMPTOMS:
        assert sym in result2["symptoms"]
        assert result2["symptoms"][sym] is not None

    # And Pydantic validation should now pass
    try:
        SymptomInfo(**result2["symptoms"])
    except ValidationError:
        pytest.fail("Validation should have passed for complete symptoms.")
