from langchain_core.messages import AIMessage

class FakeLLM:
    """
    Deterministic LLM for integration tests.
    """

    def invoke(self, messages, **kwargs):
        last = messages[-1].content.lower()

        if "itching" in last or "rash" in last:
            return AIMessage(content="You may have a skin condition. Do you have fever?")
        if "yes" in last:
            return AIMessage(content="Likely diagnosis: Allergy")
        return AIMessage(content="Need more information.")