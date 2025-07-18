from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

class ExplainerAgent:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def run(self, state):
        lab_results = state["messages"][-1].content if state["messages"] else ""
        prompt = f"Explain the lab results and implications to the patient in simple terms: '{lab_results}'"
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return {"messages": [response]}