from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from functools import lru_cache
from pathlib import Path
import os

# Get the directory where this config.py file is located
BASE_DIR = Path(__file__).resolve().parent

class Settings(BaseSettings):
    # Load from .env file in the backend directory, with fallback to system env vars
    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra="ignore")

    # Supabase (PostgreSQL DB)
    DATABASE_URL: str
    SUPABASE_URL: str
    SUPABASE_ANON_KEY: str
    
    # Backend Settings (with defaults)
    DEBUG: bool = False
    ENVIRONMENT: str = "development"
    PORT: int = 8000

    # Hugging Face
    HF_SYNC_FETCH_LIMIT: int = 3000000
    HF_APPLICABLE_LIBRARIES: list[str] = Field(default_factory=lambda: ["LiteRT", "tensorflow-lite", "tflite"])

    # Gemini API Key
    GEMINI_API_KEY: str = ""
    PIPELINE_GENERATION_MODEL: str = "gemini-2.0-flash"

    # Pipeline generation
    # Validation mode: "strict" | "loose" | "none"
    #   strict — pipeline must pass invoke() to be stored; failures mark version unsupported
    #   loose  — validation runs for shape correction; invoke() failures are stored with validation_status="not_validated"
    #   none   — skip validation entirely; pipeline is stored as-is
    PIPELINE_VALIDATION_MODE: str = "strict"
    MAX_VALIDATION_RETRIES: int = 2
    MAX_GENERATOR_WORKERS: int = 1
    MAX_VALIDATOR_DOWNLOAD_MB: int = 500
    # Maximum model file size allowed for pipeline generation (in MB).
    # Versions whose file_size_bytes exceeds this limit are skipped — no LLM call is made.
    MAX_PIPELINE_MODEL_SIZE_MB: int = 500

    # Timeouts (seconds)
    LLM_TIMEOUT_SECONDS: int = 120
    TFLITE_INVOKE_TIMEOUT_SECONDS: int = 30
    HF_FETCH_TIMEOUT_SECONDS: int = 15       # Per-chunk read timeout for small API calls
    TFLITE_DOWNLOAD_TIMEOUT_SECONDS: int = 120  # Total budget for downloading a .tflite file



# Use lru_cache so we only read the file once per startup
@lru_cache
def get_settings():
    return Settings()
    
# Usage:
# from config import get_settings
# settings = get_settings()
# print(settings.DATABASE_URL)