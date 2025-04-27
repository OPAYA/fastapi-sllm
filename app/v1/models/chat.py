from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


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


class Choice(BaseModel):
    message: Message
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatRequest(BaseModel):
    """채팅 요청 모델"""

    messages: List[Message] = Field(
        default=[Message(role="user", content="Hello, how are you?")],
        description="대화 메시지 목록",
    )
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

    class Config:
        json_schema_extra = {
            "example": {
                "messages": [{"role": "user", "content": "Hello, how are you?"}],
                "model": "gemma-1.1-2b-it",
                "temperature": 0.7,
                "max_tokens": 1000,
            }
        }


class ChatResponse(BaseModel):
    """채팅 응답 모델"""

    model: str
    choices: List[Choice]
    usage: Usage
    generation_time: float
