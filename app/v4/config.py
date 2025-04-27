from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    MODEL_PATH: Optional[str] = None
    MAX_HISTORY_LENGTH: int = 100
    PAGE_SIZE: int = 20

    class Config:
        env_file = ".env"

settings = Settings()
