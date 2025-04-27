from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import List, Optional, Dict, Any
from app.config import get_settings
import logging
import uuid
from pydantic import BaseModel, Field

from ..services.chat_history import chat_history
from ..services.chat_service import ChatService
from ..dependencies import get_chat_service

router = APIRouter()
settings = get_settings()
logger = logging.getLogger(__name__)

# 메모리 -> DB로 사용해야 하고, DB
# summarize 이부분도 페이지로 만들어야 함

class ChatMessage(BaseModel):
    role: str = Field(..., description="메시지 역할 (user/assistant)")
    content: str = Field(..., description="메시지 내용")

    class Config:
        json_schema_extra = {
            "example": {
                "role": "user",
                "content": "Hello, how are you?"
            }
        }

class ChatRequest(BaseModel):
    conversation_id: Optional[str] = Field(None, description="대화 ID (없으면 새로 생성)")
    message: str = Field(..., description="사용자 메시지")

    class Config:
        json_schema_extra = {
            "example": {
                "conversation_id": None,
                "message": "Hello, how are you?"
            }
        }

class ChatResponse(BaseModel):
    conversation_id: str = Field(..., description="대화 ID")
    response: str = Field(..., description="AI 응답")
    history: List[Dict] = Field(..., description="대화 기록")

    class Config:
        json_schema_extra = {
            "example": {
                "conversation_id": "123e4567-e89b-12d3-a456-426614174000",
                "response": "I'm doing well, thank you for asking! How can I help you today?",
                "history": [
                    {"role": "user", "content": "Hello, how are you?"},
                    {"role": "assistant", "content": "I'm doing well, thank you for asking! How can I help you today?"}
                ]
            }
        }

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service)
):
    """
    대화 기록을 유지하면서 채팅 응답을 생성합니다.

    - **conversation_id**: 대화 ID (없으면 새로 생성)
    - **message**: 사용자 메시지
    """
    # 대화 ID가 없으면 새로 생성
    conversation_id = request.conversation_id or str(uuid.uuid4())

    # 채팅 서비스를 통해 응답 생성 및 대화 기록 관리
    result = await chat_service.process_chat_with_history(
        conversation_id=conversation_id,
        message=request.message
    )

    return ChatResponse(**result)

@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """
    특정 대화의 기록을 조회합니다.

    - **conversation_id**: 대화 ID
    """
    history = chat_history.get_conversation(conversation_id)
    if not history:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"conversation_id": conversation_id, "history": history}

@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """
    특정 대화의 기록을 삭제합니다.

    - **conversation_id**: 대화 ID
    """
    chat_history.clear_conversation(conversation_id)
    return {"message": "Conversation deleted successfully"}

@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service)
):
    """
    스트리밍 방식으로 채팅 응답을 생성합니다.
    대화 기록을 유지합니다.

    - **conversation_id**: 대화 ID (없으면 새로 생성)
    - **message**: 사용자 메시지
    """
    try:
        async def generate():
            try:
                # 주입된 서비스 사용
                async for chunk in chat_service.process_chat_stream(request.dict()):
                    yield chunk
            except Exception as e:
                yield f'data: {{"error": "{str(e)}"}}\n\n'

        return StreamingResponse(generate(), media_type="text/event-stream")
    except Exception as e:
        logger.error(f"Error in chat_stream endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
