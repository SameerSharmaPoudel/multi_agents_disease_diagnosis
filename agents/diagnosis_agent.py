from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

class DiagnosisAgent:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def run(self, state):
        analysis = state["messages"][-1].content if state["messages"] else ""
        prompt = f"Based on this analysis, provide a differential diagnosis and likelihood estimation: '{analysis}'"
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return {"messages": [response]}