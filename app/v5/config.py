from pydantic import BaseModel
from typing import Dict, Any, Optional, List


class VisionModelConfig(BaseModel):
    """Vision LLM 모델 설정"""

    model_path: str = "./hyperclovax-vision-3B"
    device: str = "cpu"
    max_new_tokens: int = 8192
    temperature: float = 0.5
    top_p: float = 0.6
    repetition_penalty: float = 1.0


# 기본 설정
DEFAULT_VISION_CONFIG = VisionModelConfig()
