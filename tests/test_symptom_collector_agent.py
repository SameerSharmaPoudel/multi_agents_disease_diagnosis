import pytest
import random
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import Runnable
from pydantic import ValidationError
from typing import Any, List
from agents.symptom_collector_agent import SymptomCollectorAgent, SymptomInfo, REQUIRED_SYMPTOMS


class FakeListLLM(Runnable):
    """
    A fake LLM that returns a predefined list of responses in sequence.
    """
    def __init__(self, responses: List[str]):
        self.responses = responses
        self.index = 0

    def invoke(self, input: Any, config=None) -> AIMessage:
        if self.index >= len(self.responses):
            response = self.responses[-1]  # repeat last
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
    """Deterministic baseline test."""
    fake_llm = FakeListLLM(responses=[
        '{"fatigue": "yes", "pain": "chest"}',
        '{"fever": "high", "cough": "present", "fatigue": "yes", "pain": "chest", "duration": "2 days", "location": "chest"}'
    ])
    agent = SymptomCollectorAgent(llm=fake_llm)

    user_msg1 = "I feel tired and my chest hurts"
    state1 = {"messages": [HumanMessage(content=user_msg1)]}
    result1 = agent.run(state1)
    print_chat_transcript(1, user_msg1, result1["messages"][-1].content,
                          result1["agent_status"], result1["symptoms"])
    assert result1["agent_status"] == "incomplete"

    user_msg2 = "I also have a high fever, cough, for 2 days in my chest area"
    state2 = {"messages": result1["messages"] + [HumanMessage(content=user_msg2)]}
    result2 = agent.run(state2)
    print_chat_transcript(2, user_msg2, result2["messages"][-1].content,
                          result2["agent_status"], result2["symptoms"])
    assert result2["agent_status"] == "complete"
    SymptomInfo(**result2["symptoms"])


def test_agent_randomized_multi_turn():
    """
    Randomized multi-turn test: symptoms get filled over 2–4 turns dynamically.
    """
    # Shuffle REQUIRED_SYMPTOMS and split into 2–4 groups
    shuffled = REQUIRED_SYMPTOMS[:]
    random.shuffle(shuffled)
    num_turns = random.randint(2, 4)  # 2, 3, or 4 turns
    chunks = [shuffled[i::num_turns] for i in range(num_turns)]

    # Construct fake LLM responses progressively filling symptoms
    responses = []
    collected = {}
    for chunk in chunks:
        for sym in chunk:
            collected[sym] = f"filled_{sym}"
        responses.append(str(collected.copy()).replace("'", '"'))

    fake_llm = FakeListLLM(responses=responses)
    agent = SymptomCollectorAgent(llm=fake_llm)

    # Simulate user messages across turns
    state = {"messages": []}
    for turn, chunk in enumerate(chunks, start=1):
        user_msg = f"My symptoms are: {', '.join(chunk)}"
        state["messages"].append(HumanMessage(content=user_msg))
        result = agent.run(state)

        print_chat_transcript(turn, user_msg, result["messages"][-1].content,
                              result["agent_status"], result["symptoms"])

        if turn < num_turns:
            assert result["agent_status"] == "incomplete"
        else:
            assert result["agent_status"] == "complete"
            # all required symptoms must be filled in last turn
            for sym in REQUIRED_SYMPTOMS:
                assert sym in result["symptoms"] and result["symptoms"][sym] is not None
            SymptomInfo(**result["symptoms"])