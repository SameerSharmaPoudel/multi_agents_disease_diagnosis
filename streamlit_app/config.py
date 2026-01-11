from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional


class FrontendSettings(BaseSettings):
    fastapi_base_url: str = Field(
        "http://localhost:8000",
        env="FASTAPI_BASE_URL",
    )

    class Config:
        env_file = ".env"


settings = FrontendSettings()