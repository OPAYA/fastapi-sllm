from pydantic_settings import BaseSettings
from functools import lru_cache
import os


class Settings(BaseSettings):
    """애플리케이션 설정"""

    # SLLM 설정
    SLLM_PATH: str = "sllm"
    MODEL_PATH: str = os.path.join("models", "phi-2.q4_k_m.gguf")

    # ONNX 설정
    ONNX_MODEL_PATH: str = os.path.join("models", "onnx_qwen25.onnx")
    DEFAULT_ONNX_MODEL: str = "onnx_qwen25"

    # 기본 모델 설정
    DEFAULT_MODEL: str = "gemma-1.1-2b-it"
    DEFAULT_TEMPERATURE: float = 0.1
    DEFAULT_MAX_TOKENS: int = 200

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """설정 인스턴스를 반환합니다."""
    return Settings()
