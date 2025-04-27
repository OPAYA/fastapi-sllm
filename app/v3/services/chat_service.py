from typing import List, Dict, Any, AsyncGenerator
import json
import logging
import os
from datetime import datetime
from app.v3.models.chat import ChatRequest, ChatResponse, Message, MessageRole
from app.v3.services.sllm_service import SLLMService
from app.v3.services.prompt_service import PromptService

logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self, sllm_service: SLLMService):
        """
        ChatService 생성자

        Args:
            sllm_service: 의존성 주입을 통해 제공되는 SLLMService 인스턴스
        """
        self.sllm_service = sllm_service
        self.prompt_service = PromptService()
        logger.info("ChatService initialized")

    async def process_chat(self, request: ChatRequest) -> ChatResponse:
        """
        채팅 요청을 처리하고 응답을 생성합니다.

        Args:
            request: 채팅 요청

        Returns:
            ChatResponse: 생성된 응답
        """
        try:
            # 프롬프트 템플릿 ID가 있으면 해당 템플릿 사용
            template_id = request.metadata.get("template_id") if request.metadata else None

            if template_id:
                # 메시지 목록을 문자열로 변환
                messages_text = self._format_messages(request.messages)

                # 프롬프트 템플릿 렌더링
                prompt_response = self.prompt_service.render_prompt(
                    template_id=template_id,
                    variables={"messages": messages_text}
                )

                # 렌더링된 프롬프트를 시스템 메시지로 추가
                system_message = Message(
                    role=MessageRole.SYSTEM,
                    content=prompt_response.rendered_prompt,
                    timestamp=datetime.utcnow(),
                    metadata=None  # 명시적으로 None으로 설정
                )

                # 시스템 메시지를 첫 번째로 추가
                messages = [system_message] + request.messages
            else:
                # 템플릿 ID가 없으면 기본 메시지 사용
                messages = request.messages

            # LLM 호출
            response = await self.sllm_service.generate_response(
                messages=messages,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p,
                frequency_penalty=request.frequency_penalty,
                presence_penalty=request.presence_penalty
            )

            # sllm_service.generate_response의 응답을 그대로 반환
            return response
        except Exception as e:
            logger.error(f"Error in process_chat: {str(e)}")
            raise

    async def process_chat_stream(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        """
        스트리밍 방식으로 채팅 응답을 생성합니다.

        Args:
            request: 채팅 요청

        Yields:
            str: 생성된 응답 청크
        """
        try:
            # 프롬프트 템플릿 ID가 있으면 해당 템플릿 사용
            template_id = request.metadata.get("template_id") if request.metadata else None

            if template_id:
                # 메시지 목록을 문자열로 변환
                messages_text = self._format_messages(request.messages)

                # 프롬프트 템플릿 렌더링
                prompt_response = self.prompt_service.render_prompt(
                    template_id=template_id,
                    variables={"messages": messages_text}
                )

                # 렌더링된 프롬프트를 시스템 메시지로 추가
                system_message = Message(
                    role=MessageRole.SYSTEM,
                    content=prompt_response.rendered_prompt,
                    timestamp=datetime.utcnow()
                )

                # 시스템 메시지를 첫 번째로 추가
                messages = [system_message] + request.messages
            else:
                # 템플릿 ID가 없으면 기본 메시지 사용
                messages = request.messages

            # 스트리밍 응답 생성
            async for chunk in self.sllm_service.generate_streaming_response(
                messages=messages,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p,
                frequency_penalty=request.frequency_penalty,
                presence_penalty=request.presence_penalty
            ):
                # 청크가 딕셔너리인 경우 content 필드 추출
                if isinstance(chunk, dict):
                    content = chunk.get("content", "")
                else:
                    content = str(chunk)

                yield f'data: {{"text": "{content}"}}\n\n'

            # 응답 완료 신호
            yield 'data: {"done": true}\n\n'
        except Exception as e:
            logger.error(f"Error in process_chat_stream: {str(e)}")
            yield f'data: {{"error": "{str(e)}"}}\n\n'

    def _format_messages(self, messages: List[Message]) -> str:
        """
        메시지 목록을 문자열로 변환합니다.

        Args:
            messages: 메시지 목록

        Returns:
            str: 변환된 메시지 문자열
        """
        formatted_messages = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                formatted_messages.append(f"System: {msg.content}")
            elif msg.role == MessageRole.USER:
                formatted_messages.append(f"Human: {msg.content}")
            elif msg.role == MessageRole.ASSISTANT:
                formatted_messages.append(f"Assistant: {msg.content}")

        return "\n\n".join(formatted_messages)
