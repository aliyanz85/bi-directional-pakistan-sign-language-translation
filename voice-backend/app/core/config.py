from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Application settings and configuration."""

    # API Settings
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "D-VOICE Animation Service"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "Backend service for PSL sign language animations"

    # CORS Settings - Allow frontend to call this API
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",  # Vite default dev server
        "http://localhost:3000",  # Alternative dev port
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ]

    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Animation Settings
    ANIMATIONS_CONFIG_PATH: str = "app/animations_config.json"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
