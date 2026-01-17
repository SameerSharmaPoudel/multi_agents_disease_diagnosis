from pydantic_settings import BaseSettings
from pydantic import Field


class FrontendSettings(BaseSettings):
    app_env: str = Field("dev", env="APP_ENV")

    fastapi_base_url: str = Field(
        "http://localhost:8000",
        env="FASTAPI_BASE_URL",
        description="Base URL of the FastAPI backend"
    )

    class Config:
        env_file = ".env.frontend"
        extra = "forbid"


frontend_settings = FrontendSettings()