from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from config import settings


class ModelLoader:
    def __init__(self, provider: str, model: str):
        self.provider = provider
        self.model = model

    def load_llm(self):
        if self.provider == "groq":
            if not settings.groq_api_key:
                raise RuntimeError("GROQ_API_KEY not set")
            return ChatGroq(
                model=self.model,
                api_key=settings.groq_api_key,
            )

        if self.provider == "openai":
            if not settings.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY not set")
            return ChatOpenAI(
                model=self.model,
                api_key=settings.openai_api_key,
            )

        raise ValueError(f"Unsupported LLM provider: {self.provider}")