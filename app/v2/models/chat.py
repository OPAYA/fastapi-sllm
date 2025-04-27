from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from enum import Enum


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "role": "user",
                "content": "안녕하세요!",
                "timestamp": "2024-04-19T12:00:00Z",
            }
        }


class ChatRequest(BaseModel):
    messages: List[Message]
    model: str = "sllm"
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)

    class Config:
        json_schema_extra = {
            "example": {
                "messages": [
                    {
                        "role": "user",
                        "content": "안녕하세요!",
                        "timestamp": "2024-04-19T12:00:00Z",
                    }
                ],
                "model": "sllm",
                "temperature": 0.7,
                "max_tokens": 100,
            }
        }


class ChatResponse(BaseModel):
    conversation_id: str
    messages: List[Message]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "conversation_id": "conv_123",
                "messages": [
                    {
                        "role": "user",
                        "content": "안녕하세요!",
                        "timestamp": "2024-04-19T12:00:00Z",
                    },
                    {
                        "role": "assistant",
                        "content": "안녕하세요! 무엇을 도와드릴까요?",
                        "timestamp": "2024-04-19T12:00:01Z",
                    },
                ],
                "created_at": "2024-04-19T12:00:00Z",
                "updated_at": "2024-04-19T12:00:01Z",
            }
        }


class StreamingChatResponse(BaseModel):
    conversation_id: str
    content: str
    is_complete: bool = False

    class Config:
        json_schema_extra = {
            "example": {
                "conversation_id": "conv_123",
                "content": "안녕하세요!",
                "is_complete": False,
            }
        }
