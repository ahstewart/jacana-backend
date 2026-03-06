from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from functools import lru_cache
from pathlib import Path
import os

# Get the directory where this config.py file is located
BASE_DIR = Path(__file__).resolve().parent

class Settings(BaseSettings):
    # Load from .env file in the backend directory, with fallback to system env vars
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # 1. Define the Keys you need (Required)
    DATABASE_URL: str
    SUPABASE_DB_URL: str
    SUPABASE_DB_PASSWORD: str

    # Authentication
    SUPABASE_JWT_SECRET: str

    # Admin Tool (Optional - for Storage/Auth management)
    SUPABASE_URL: str
    SUPABASE_SERVICE_KEY: str
    SUPABASE_PUBLISHABLE_API_KEY: str
    
    # Backend Settings (with defaults)
    DEBUG: bool = False
    ENVIRONMENT: str = "production"
    PORT: int = 8000

    # Hugging Face
    HF_SYNC_FETCH_LIMIT: int = 3000000
    HF_APPLICABLE_LIBRARIES: list[str] = Field(default_factory=lambda: ["LiteRT", "tensorflow-lite", "tflite"])

    # Gemini API Key
    GEMINI_API_KEY: str = ""
    PIPELINE_GENERATION_MODEL: str = "gemini-2.0-flash"



# Use lru_cache so we only read the file once per startup
@lru_cache
def get_settings():
    try:
        return Settings()
    except Exception as e:
        # Provide detailed error message
        env_file = BASE_DIR / ".env"
        if not env_file.exists():
            raise FileNotFoundError(
                f".env file not found at {env_file}. "
                f"Please create the file or set environment variables."
            )
        raise ValueError(
            f"Failed to load configuration from .env file: {e}\n"
            f"Looking for .env at: {env_file}\n"
            f"Make sure all required fields are present in .env file."
        )

# Usage:
# from config import get_settings
# settings = get_settings()
# print(settings.DATABASE_URL)