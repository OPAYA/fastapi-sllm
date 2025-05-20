import logging
from fastapi import Depends
from app.provider.vision_provider import (
    get_vision_tokenizer,
    get_vision_processor,
    get_vision_model,
    get_vision_model_info,
)
from app.v5.services.vision_service import VisionService

logger = logging.getLogger(__name__)


def get_vision_service(
    tokenizer=Depends(get_vision_tokenizer),
    processor=Depends(get_vision_processor),
    model=Depends(get_vision_model),
) -> VisionService:
    """
    Vision LLM 서비스 의존성 주입 함수
    토크나이저, 프로세서, 모델 인스턴스를 주입합니다.
    """
    logger.info("VisionService 의존성 주입")
    return VisionService(tokenizer=tokenizer, processor=processor, model=model)
