from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union, Literal


class ImageContent(BaseModel):
    """이미지 콘텐츠 모델"""

    type: Literal["image"] = "image"
    filename: str = Field(..., description="이미지 파일명")
    image: str = Field(..., description="이미지 URL 또는 Base64 인코딩된 이미지 데이터")
    ocr: Optional[str] = Field(None, description="OCR 텍스트")
    lens_keywords: Optional[str] = Field(None, description="렌즈 키워드")
    lens_local_keywords: Optional[str] = Field(None, description="로컬 렌즈 키워드")


class TextContent(BaseModel):
    """텍스트 콘텐츠 모델"""

    type: Literal["text"] = "text"
    text: str = Field(..., description="텍스트 내용")


class VideoContent(BaseModel):
    """비디오 콘텐츠 모델"""

    type: Literal["video"] = "video"
    filename: str = Field(..., description="비디오 파일명")
    video: str = Field(..., description="비디오 URL 또는 Base64 인코딩된 비디오 데이터")


ContentType = Union[TextContent, ImageContent, VideoContent]


class VisionChatMessage(BaseModel):
    """비전 채팅 메시지 모델"""

    role: Literal["system", "user", "assistant"] = Field(..., description="메시지 역할")
    content: Union[ContentType, List[ContentType]] = Field(..., description="메시지 내용")


class VisionChatRequest(BaseModel):
    """비전 채팅 요청 모델"""

    messages: List[VisionChatMessage] = Field(..., description="채팅 메시지 목록")
    temperature: Optional[float] = Field(0.5, description="생성 다양성 조절 파라미터")
    top_p: Optional[float] = Field(0.6, description="생성 다양성 조절 파라미터")
    max_new_tokens: Optional[int] = Field(8192, description="최대 생성 토큰 수")
    repetition_penalty: Optional[float] = Field(1.0, description="반복 패널티")

    class Config:
        json_schema_extra = {
            "example": {
                "messages": [
                    {
                        "role": "system",
                        "content": {"type": "text", "text": "System Prompt"},
                    },
                    {"role": "user", "content": {"type": "text", "text": "User Text"}},
                    {
                        "role": "user",
                        "content": {
                            "type": "image",
                            "filename": "example.png",
                            "image": "https://example.com/image.png",
                            "ocr": "OCR text from the image",
                        },
                    },
                ],
                "temperature": 0.5,
                "top_p": 0.6,
                "max_new_tokens": 1024,
                "repetition_penalty": 1.0,
            }
        }


class VisionChatResponse(BaseModel):
    """비전 채팅 응답 모델"""

    response: str = Field(..., description="응답 텍스트")
    usage: Dict[str, Any] = Field(..., description="토큰 사용량 정보")
    metadata: Dict[str, Any] = Field(..., description="모델 메타데이터")
