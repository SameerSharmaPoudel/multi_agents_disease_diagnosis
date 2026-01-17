from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class BackendSettings(BaseSettings):
    # Environment
    app_env: str = Field("dev", env="APP_ENV")

    # LLM configuration
    llm_provider: str = Field("groq", env="LLM_PROVIDER")
    llm_model: str = Field("llama3-70b-8192", env="LLM_MODEL")

    llm_fallback_provider: Optional[str] = Field(None, env="LLM_FALLBACK_PROVIDER")
    llm_fallback_model: Optional[str] = Field(None, env="LLM_FALLBACK_MODEL")

    # Secrets
    groq_api_key: Optional[str] = Field(None, env="GROQ_API_KEY")
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")

    class Config:
        env_file = ".env.backend"
        extra = "forbid"


backend_settings = BackendSettings()