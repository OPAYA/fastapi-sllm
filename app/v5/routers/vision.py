from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
import logging
from app.provider.vision_provider import get_vision_model_info
from app.v5.dependencies import get_vision_service
from app.v5.models.vision import VisionChatRequest, VisionChatResponse
from app.v5.services.vision_service import VisionService

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/vision/chat", response_model=VisionChatResponse)
async def vision_chat(
    request: VisionChatRequest,
    vision_service: VisionService = Depends(get_vision_service),
):
    """
    HyperCloVAX Vision 모델을 사용하여 이미지, 비디오와 함께 채팅 응답을 생성합니다.

    - **messages**: 채팅 메시지 목록 (텍스트, 이미지, 비디오 포함 가능)
    - **temperature**: 생성 다양성 조절 파라미터 (기본값: 0.5)
    - **top_p**: 생성 다양성 조절 파라미터 (기본값: 0.6)
    - **max_new_tokens**: 최대 생성 토큰 수 (기본값: 8192)
    - **repetition_penalty**: 반복 패널티 (기본값: 1.0)
    """
    try:
        response = await vision_service.generate_response(request)
        return response
    except Exception as e:
        logger.error(f"Vision 채팅 엔드포인트 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/vision/info")
async def vision_model_info():
    """
    HyperCloVAX Vision 모델 정보를 반환합니다.
    """
    model_info = get_vision_model_info()
    return {
        "model": "hyperclovax-vision-3B",
        "status": "loaded" if model_info["model_loaded"] else "not_loaded",
        "model_path": model_info["model_path"],
        "load_time": model_info["model_load_time"],
        "device": model_info["device"],
    }
