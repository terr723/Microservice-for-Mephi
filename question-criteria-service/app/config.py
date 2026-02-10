from pydantic_settings import BaseSettings
from typing import Literal
from functools import lru_cache

class Settings(BaseSettings):
    EMBEDDING_MODEL_NAME: str = "intfloat/multilingual-e5-small"
    MODEL_DEVICE: Literal["cpu", "cuda", "mps"] = "cpu"
    ENVIRONMENT: Literal["development", "production", "testing"] = "development"
    DEBUG: bool = False

    class Config:
        env_file = ".env"
        case_sensitive = False

@lru_cache()
def get_settings() -> Settings:
    return Settings()