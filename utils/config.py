from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Environment
    app_env: str = Field("dev", env="APP_ENV")

    # Primary model
    llm_provider: str = Field("groq", env="LLM_PROVIDER")
    llm_model: str = Field("llama3-70b-8192", env="LLM_MODEL")

    # Fallback model (optional)
    llm_fallback_provider: Optional[str] = Field(None, env="LLM_FALLBACK_PROVIDER")
    llm_fallback_model: Optional[str] = Field(None, env="LLM_FALLBACK_MODEL")

    # API keys
    groq_api_key: Optional[str] = Field(None, env="GROQ_API_KEY")
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")

    class Config:
        env_file = ".env"


settings = Settings()

