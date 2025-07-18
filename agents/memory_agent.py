from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

class MemoryAgent:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def run(self, state):
        final_output = state["messages"][-1].content if state["messages"] else ""
        prompt = f"Summarize and store the patient's visit and diagnosis notes: '{final_output}'"
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return {"messages": [response]}