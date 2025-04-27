from typing import List, AsyncGenerator, Dict, Any, Union, Optional
from app.v4.services.sllm_service import SLLMService
import uuid
import json
import logging
from ..services.chat_history import chat_history

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(
        self,
        sllm_service: SLLMService,
    ):
        """
        ChatService 생성자

        Args:
            sllm_service: 의존성 주입을 통해 제공되는 SLLMService 인스턴스
        """
        self.sllm_service = sllm_service

    async def process_chat(self, conversation_id: str, message: str) -> str:
        # 대화 기록 가져오기
        history = chat_history.get_conversation(conversation_id)

        # 메시지 포맷 변환
        messages = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in history
        ]

        # 현재 메시지 추가
        messages.append({"role": "user", "content": message})

        # LLM 호출
        response = await self.sllm_service.generate_response(messages)

        return response

    async def process_chat_with_history(self, conversation_id: str, message: str) -> Dict:
        # 사용자 메시지 기록
        chat_history.add_message(conversation_id, "user", message)

        # LLM 응답 생성
        response = await self.process_chat(conversation_id, message)

        # AI 응답 기록
        chat_history.add_message(conversation_id, "assistant", response)

        # 전체 대화 기록 반환
        history = chat_history.get_conversation(conversation_id)

        return {
            "conversation_id": conversation_id,
            "response": response,
            "history": history
        }

    async def process_chat_stream(
        self, request: Dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """스트리밍 방식으로 채팅 응답을 생성합니다."""
        try:
            # 대화 ID가 없으면 새로 생성
            conversation_id = request.get("conversation_id") or str(uuid.uuid4())
            message = request.get("message", "")

            # 사용자 메시지 기록
            chat_history.add_message(conversation_id, "user", message)

            # 대화 기록 가져오기
            history = chat_history.get_conversation(conversation_id)

            # 메시지 포맷 변환
            messages = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in history
            ]

            # 스트리밍 응답 생성
            async for chunk in self.sllm_service.generate_streaming_response(messages):
                yield f'data: {{"text": "{chunk}"}}\n\n'

            # 응답 완료 후 AI 응답 기록
            full_response = "".join([chunk for chunk in self.sllm_service.last_response])
            chat_history.add_message(conversation_id, "assistant", full_response)

            # 응답 완료 신호
            yield 'data: {"done": true}\n\n'

        except Exception as e:
            logger.error(f"Error in process_chat_stream: {str(e)}")
            yield f'data: {{"error": "{str(e)}"}}\n\n'
