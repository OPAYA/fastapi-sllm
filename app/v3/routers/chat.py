from enum import Enum
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from typing import List, Optional
from pydantic import BaseModel, Field
import logging

from ..services.chat_service import ChatService
from ..dependencies import get_chat_service

router = APIRouter()
logger = logging.getLogger(__name__)

class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    """메시지 모델"""

    role: MessageRole
    content: str = Field(..., description="메시지 내용")
    timestamp: Optional[str] = Field(None, description="메시지 생성 시간")

    def dict(self, *args, **kwargs):
        """JSON 직렬화를 위한 dict 메서드 오버라이드"""
        d = super().dict(*args, **kwargs)
        if d.get("timestamp") is None:
            d["timestamp"] = datetime.utcnow().isoformat()
        return d


class ChatRequest(BaseModel):
    """채팅 요청 모델"""

    messages: List[Message] = Field(
        default=[Message(role=MessageRole.USER, content="Hello, how are you?")],
        description="대화 메시지 목록",
    )
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    metadata: Optional[dict] = Field(None, description="추가 메타데이터 (예: template_id)")

    class Config:
        json_schema_extra = {
            "example": {
                "messages": [{"role": "user", "content": "Hello, how are you?"}],
                "model": "gemma-1.1-2b-it",
                "temperature": 0.7,
                "max_tokens": 1000,
                "metadata": {"template_id": "code_assistant"}
            }
        }


class Choice(BaseModel):
    message: Message
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    """채팅 응답 모델"""

    model: str
    choices: List[Choice]
    usage: Usage
    generation_time: float

    class Config:
        json_schema_extra = {
            "example": {
                "model": "gemma-1.1-2b-it",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "I'm doing well, thank you for asking! How can I help you today?",
                            "timestamp": "2024-04-27T12:00:00.000Z"
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30
                },
                "generation_time": 0.5
            }
        }


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service)
):
    """
    채팅 메시지를 처리하고 응답을 생성합니다.
    """
    try:
        response = await chat_service.process_chat(request)

        # ChatResponse 형식으로 변환
        return ChatResponse(
            model=request.model or "sllm",
            choices=[
                Choice(
                    message=Message(
                        role=MessageRole.ASSISTANT,
                        content=response.messages[-1].content
                    ),
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=response.usage.get("prompt_tokens", 0),
                completion_tokens=response.usage.get("completion_tokens", 0),
                total_tokens=response.usage.get("total_tokens", 0)
            ),
            generation_time=response.metadata.get("inference_time", 0.0) if response.metadata else 0.0
        )
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service)
):
    """
    스트리밍 방식으로 채팅 메시지를 처리하고 응답을 생성합니다.
    """
    try:
        return StreamingResponse(
            chat_service.process_chat_stream(request),
            media_type="text/event-stream"
        )
    except Exception as e:
        logger.error(f"Error in chat_stream endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
