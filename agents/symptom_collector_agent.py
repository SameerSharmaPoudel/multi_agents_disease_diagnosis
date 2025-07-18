from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage


class SymptomCollectorAgent:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def run(self, state):
        user_input = state["messages"][-1].content if state["messages"] else ""
        prompt = f"You are a patient interviewer. Collect structured symptom information from the following user input: '{user_input}'"
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return {"messages": [response]}