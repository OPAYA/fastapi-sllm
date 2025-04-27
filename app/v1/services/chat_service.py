from typing import List
from app.v1.models.chat import ChatRequest, ChatResponse
from app.v1.services.sllm_service import SLLMService
import logging

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self, sllm_service: SLLMService):
        """
        ChatService 생성자

        Args:
            sllm_service: 의존성 주입을 통해 제공되는 SLLMService 인스턴스
        """
        # 항상 의존성 주입을 통해 제공 - 의존성 주입이 누락된 경우 에러가 발생하도록 함
        self.sllm_service = sllm_service

    async def process_chat(self, request: ChatRequest) -> ChatResponse:
        """채팅 요청을 처리하고 응답을 생성합니다."""
        response = await self.sllm_service.generate_response(
            messages=request.messages,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        return response

    async def get_available_models(self) -> List[str]:
        """사용 가능한 모델 목록을 조회합니다."""
        models = await self.sllm_service.get_available_models()
        return models
