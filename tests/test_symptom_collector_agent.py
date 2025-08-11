import pytest
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import ValidationError
from langchain_core.runnables import Runnable
from langchain_core.outputs import LLMResult
from typing import Any, List
from agents.symptom_collector_agent import SymptomCollectorAgent, SymptomInfo, REQUIRED_SYMPTOMS

class FakeListLLM(Runnable):
    """
    A fake LLM that returns a predefined list of responses in sequence.
    Compatible with LangChain's Runnable interface so it can be used in `|` chains.
    """
    def __init__(self, responses: List[str]):
        self.responses = responses
        self.index = 0

    def invoke(self, input: Any, config=None) -> AIMessage:
        if self.index >= len(self.responses):
            # Instead of raising, return last response repeatedly
            response = self.responses[-1]
        else:
            response = self.responses[self.index]
            self.index += 1
        return AIMessage(content=response)

    async def ainvoke(self, input: Any, config=None) -> AIMessage:
        return self.invoke(input, config)

def print_chat_transcript(turn_num, user_msg, agent_msg, status, symptoms):
    print(f"\n--- TURN {turn_num} ---")
    print(f"[USER]  {user_msg}")
    print(f"[AGENT] {agent_msg}")
    print(f"[STATUS] {status}")
    if symptoms:
        print(f"[SYMPTOMS] {symptoms}")

def test_agent_multi_turn_completion():

    fake_llm = FakeListLLM(responses=[
        '{"fatigue": "yes", "pain": "chest"}',  
        '{"fever": "high", "cough": "present", "fatigue": "yes", "pain": "chest", "duration": "2 days", "location": "chest"}'
    ])

    agent = SymptomCollectorAgent(llm=fake_llm)

    # First user message (incomplete symptoms)
    user_msg1 = "I feel tired and my chest hurts"
    state1 = {"messages": [HumanMessage(content=user_msg1)]}
    result1 = agent.run(state1)

    print_chat_transcript(
        turn_num=1,
        user_msg=user_msg1,
        agent_msg=result1["messages"][-1].content,
        status=result1["agent_status"],
        symptoms=result1["symptoms"]
    )

    assert result1["agent_status"] == "incomplete"
    assert result1["symptoms"] == {}
    assert isinstance(result1["messages"][-1], AIMessage)

    # Second user message (completing symptoms)
    user_msg2 = "I also have a high fever, cough, for 2 days in my chest area"
    state2 = {"messages": result1["messages"] + [HumanMessage(content=user_msg2)]}
    result2 = agent.run(state2)

    print_chat_transcript(
        turn_num=2,
        user_msg=user_msg2,
        agent_msg="Thanks! I now have all your symptoms recorded.",
        status=result2["agent_status"],
        symptoms=result2["symptoms"]
    )

    assert result2["agent_status"] == "complete"
    for sym in REQUIRED_SYMPTOMS:
        assert sym in result2["symptoms"]
        assert result2["symptoms"][sym] is not None

    try:
        SymptomInfo(**result2["symptoms"])
    except ValidationError:
        pytest.fail("Validation should have passed for complete symptoms.")
