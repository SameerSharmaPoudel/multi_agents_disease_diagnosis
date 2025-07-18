from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

class SymptomAnalyzerAgent:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def run(self, state):
        symptoms = state["messages"][-1].content if state["messages"] else ""
        prompt = f"Analyze the symptoms and infer potential disease categories: '{symptoms}'"
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return {"messages": [response]}