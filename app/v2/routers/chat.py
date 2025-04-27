from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import List, Optional
from app.v2.models.chat import Message, ChatRequest, ChatResponse, StreamingChatResponse
from app.v2.services.chat_service import ChatService
from app.v2.dependencies import get_chat_service
from app.config import get_settings
import logging

router = APIRouter()
settings = get_settings()
logger = logging.getLogger(__name__)


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest, chat_service: ChatService = Depends(get_chat_service)
):
    """
    채팅 요청을 처리하고 응답을 생성합니다.

    - **messages**: 대화 메시지 목록
    - **model**: 사용할 모델 (기본값: sllm)
    - **temperature**: 응답의 무작위성 (0.0 ~ 1.0)
    - **max_tokens**: 최대 토큰 수
    """
    try:
        # 주입된 서비스 사용
        response = await chat_service.process_chat(request)

        return response
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest, chat_service: ChatService = Depends(get_chat_service)
):
    """
    스트리밍 방식으로 채팅 응답을 생성합니다.

    - **messages**: 대화 메시지 목록
    - **model**: 사용할 모델 (기본값: sllm)
    - **temperature**: 응답의 무작위성 (0.0 ~ 1.0)
    - **max_tokens**: 최대 토큰 수
    """
    try:

        async def generate():
            try:
                # 주입된 서비스 사용
                async for chunk in chat_service.process_chat_stream(request):
                    yield chunk
            except Exception as e:
                yield f'data: {{"error": "{str(e)}"}}\n\n'

        return StreamingResponse(generate(), media_type="text/event-stream")
    except Exception as e:
        logger.error(f"Error in chat_stream endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
