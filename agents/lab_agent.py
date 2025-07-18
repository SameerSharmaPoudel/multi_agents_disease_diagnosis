from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

class LabAgent:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def run(self, state):
        diagnosis = state["messages"][-1].content if state["messages"] else ""
        prompt = f"Suggest lab tests to confirm or refute the diagnosis: '{diagnosis}'"
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return {"messages": [response]}