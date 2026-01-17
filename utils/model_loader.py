from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from utils.config_backend import backend_settings


class ModelLoader:
    def __init__(self, provider: str, model: str):
        self.provider = provider
        self.model = model

    def load_llm(self):
        if self.provider == "groq":
            if not backend_settings.groq_api_key:
                raise RuntimeError("GROQ_API_KEY not set")
            return ChatGroq(
                model=self.model,
                api_key=backend_settings.groq_api_key,
            )

        if self.provider == "openai":
            if not backend_settings.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY not set")
            return ChatOpenAI(
                model=self.model,
                api_key=backend_settings.openai_api_key,
            )

        raise ValueError(f"Unsupported LLM provider: {self.provider}")