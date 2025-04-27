from typing import List, AsyncGenerator
from app.v2.models.chat import ChatRequest, ChatResponse
from app.v2.services.sllm_service import SLLMService
import json
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

    async def process_chat_stream(
        self, request: ChatRequest
    ) -> AsyncGenerator[str, None]:
        """스트리밍 방식으로 채팅 응답을 생성합니다."""
        try:
            response = await self.sllm_service.generate_response(
                messages=request.messages,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )

            # 응답을 청크로 나누어 전송
            content = response.messages[-1].content
            chunk_size = 10  # 한 번에 전송할 문자 수

            for i in range(0, len(content), chunk_size):
                chunk = content[i : i + chunk_size]
                yield f"data: {json.dumps({'content': chunk, 'is_complete': False})}\n\n"

            # 완료 메시지 전송
            yield f"data: {json.dumps({'content': '', 'is_complete': True})}\n\n"

        except Exception as e:
            logger.error(f"Error in process_chat_stream: {str(e)}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
