from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
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
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "role": "user",
                "content": "Hello!",
                "timestamp": "2024-04-19T12:00:00Z",
                "metadata": {"language": "en"},
            }
        }


class ChatRequest(BaseModel):
    messages: List[Message]
    model: str = "sllm"
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="추가 메타데이터 (예: template_id)")

    class Config:
        json_schema_extra = {
            "example": {
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello!",
                        "timestamp": "2024-04-19T12:00:00Z",
                        "metadata": {"language": "en"},
                    }
                ],
                "model": "sllm",
                "temperature": 0.7,
                "max_tokens": 100,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "metadata": {"template_id": "default_chat"}
            }
        }


class ChatResponse(BaseModel):
    conversation_id: str
    messages: List[Message]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    usage: Dict[str, int]
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "conversation_id": "conv_123",
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello!",
                        "timestamp": "2024-04-19T12:00:00Z",
                        "metadata": {"language": "ko"},
                    },
                    {
                        "role": "assistant",
                        "content": "Hello! How can I help you?",
                        "timestamp": "2024-04-19T12:00:01Z",
                        "metadata": {"language": "en"},
                    },
                ],
                "created_at": "2024-04-19T12:00:00Z",
                "updated_at": "2024-04-19T12:00:01Z",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                },
                "metadata": {
                    "model_version": "1.0",
                    "template_id": "default_chat",
                    "template_version": "1.0"
                },
            }
        }


class StreamingChatResponse(BaseModel):
    conversation_id: str
    content: str
    is_complete: bool = False
    usage: Optional[Dict[str, int]] = None
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "conversation_id": "conv_123",
                "content": "Hello!",
                "is_complete": False,
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
                "metadata": {"model_version": "1.0"},
            }
        }
