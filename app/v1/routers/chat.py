from fastapi import APIRouter, HTTPException, Depends
from typing import List
from app.v1.models.chat import ChatRequest, ChatResponse
from app.v1.services.sllm_service import SLLMService
from app.config import get_settings

router = APIRouter()

# 설정 가져오기
settings = get_settings()

def get_chat_service():
    """채팅 서비스 의존성 주입"""
    return SLLMService()

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    chat_service: SLLMService = Depends(get_chat_service)
):
    """채팅 요청을 처리하고 응답을 생성합니다."""
    try:
        return await chat_service.generate_response(
            messages=request.messages,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models", response_model=List[str])
async def list_models(
    chat_service: SLLMService = Depends(get_chat_service)
):
    """사용 가능한 모델 목록을 조회합니다."""
    try:
        return await chat_service.get_available_models()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/default-model")
async def get_default_model():
    """기본 모델 정보를 반환합니다."""
    return {
        "model": settings.DEFAULT_MODEL,
        "temperature": settings.DEFAULT_TEMPERATURE,
        "max_tokens": settings.DEFAULT_MAX_TOKENS,
    }
