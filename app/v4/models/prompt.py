from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List


class PromptTemplate(BaseModel):
    """프롬프트 템플릿 모델"""
    id: str
    name: str
    description: Optional[str] = None
    template: str
    version: str = "1.0"
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        schema_extra = {
            "example": {
                "id": "default_chat",
                "name": "기본 채팅 프롬프트",
                "description": "일반적인 대화를 위한 기본 프롬프트 템플릿",
                "template": "You are a helpful AI assistant. Please respond directly and concisely to the user's questions.\n\n{{messages}}",
                "version": "1.0",
                "tags": ["chat", "general"],
                "metadata": {"author": "SSAFY Team", "created_at": "2023-05-30"}
            }
        }


class PromptVariable(BaseModel):
    """프롬프트 변수 모델"""
    name: str
    value: Any

    class Config:
        schema_extra = {
            "example": {
                "name": "context",
                "value": "Paris is the capital of France."
            }
        }


class PromptRequest(BaseModel):
    """프롬프트 렌더링 요청"""
    template_id: str
    variables: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        schema_extra = {
            "example": {
                "template_id": "default_chat",
                "variables": {
                    "messages": "Human: What is the capital of France?\n\nAssistant:"
                }
            }
        }


class PromptResponse(BaseModel):
    """프롬프트 렌더링 응답"""
    rendered_prompt: str
    template_id: str
    variables: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)
