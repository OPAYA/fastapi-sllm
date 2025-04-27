# 이 파일은 이제 사용되지 않습니다.
# app.services.sllm_provider에서 제공하는 중앙 SLLM 서비스를 대신 사용하세요.

import os
import logging
import time
import uuid
from typing import List, Optional, Dict
from datetime import datetime
from llama_cpp import Llama
from app.v2.models.chat import Message, ChatResponse, MessageRole
from app.config import get_settings
from app.provider.sllm_provider import get_llama_model, get_model_info

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SLLMService:
    """대화 기록을 관리하는 채팅 서비스"""

    def __init__(self):
        settings = get_settings()
        self.default_model = settings.DEFAULT_MODEL
        self.default_temperature = settings.DEFAULT_TEMPERATURE
        self.default_max_tokens = settings.DEFAULT_MAX_TOKENS
        self.conversations: Dict[str, List[Message]] = {}

        # 중앙 관리되는 모델 공유
        self._model_info = get_model_info()

    @property
    def model(self) -> Optional[Llama]:
        """모델 인스턴스를 가져옵니다."""
        return get_llama_model()

    @property
    def model_loaded(self) -> bool:
        """모델이 로드되었는지 확인합니다."""
        return self._model_info["model_loaded"]

    @property
    def model_load_time(self) -> float:
        """모델 로드 시간을 반환합니다."""
        return self._model_info["model_load_time"]

    def _get_or_create_conversation(self, conversation_id: str) -> List[Message]:
        """대화 기록을 가져오거나 새로 생성합니다."""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        return self.conversations[conversation_id]

    async def generate_response(
        self,
        messages: List[Message],
        conversation_id: Optional[str] = None,
        model: str = None,
        temperature: float = None,
        max_tokens: Optional[int] = None,
    ) -> ChatResponse:
        """Llama 모델을 사용하여 응답을 생성하고 대화 기록을 관리합니다.

        Args:
            messages (List[Message]): 대화 메시지 목록
            conversation_id (Optional[str], optional): 대화 ID
            model (str, optional): 사용할 모델 이름
            temperature (float, optional): 생성 다양성 조절 파라미터
            max_tokens (Optional[int], optional): 최대 생성 토큰 수

        Returns:
            ChatResponse: 생성된 응답
        """
        logger.info(
            f"generate_response 호출됨: {messages}, {conversation_id}, {model}, {temperature}, {max_tokens}"
        )

        # 기본값 설정
        model = model or self.default_model
        temperature = (
            temperature if temperature is not None else self.default_temperature
        )
        max_tokens = max_tokens or self.default_max_tokens
        conversation_id = conversation_id or str(uuid.uuid4())

        # 대화 기록 가져오기
        conversation = self._get_or_create_conversation(conversation_id)
        conversation.extend(messages)

        # 중앙 관리되는 모델이 로드되지 않은 경우 테스트 모드로 실행
        if not self.model_loaded or self.model is None:
            logger.warning("모델이 로드되지 않았습니다. 테스트 모드로 실행됩니다.")
            return self._generate_test_response(conversation, conversation_id)

        # 프롬프트 구성
        prompt = "You are a helpful AI assistant. Please respond directly and concisely to the user's questions.\n\n"

        for msg in conversation:
            if msg.role == MessageRole.SYSTEM:
                prompt += f"System: {msg.content}\n\n"
            elif msg.role == MessageRole.USER:
                prompt += f"Human: {msg.content}\n\n"
            elif msg.role == MessageRole.ASSISTANT:
                prompt += f"Assistant: {msg.content}\n\n"

        prompt += "Assistant: "

        logger.info(f"프롬프트: {prompt}")

        try:
            start_time = time.time()
            response = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["Human:", "System:", "Assistant:"],
            )
            inference_time = time.time() - start_time
            logger.info(f"추론 시간: {inference_time:.2f}초")

            generated_text = response["choices"][0]["text"].strip()
            logger.info(f"모델 응답: {generated_text}")

            if not generated_text:
                generated_text = "죄송합니다. 응답을 생성할 수 없습니다. 다시 시도해주세요."

            # 새로운 메시지 생성
            assistant_message = Message(
                role=MessageRole.ASSISTANT,
                content=generated_text,
                timestamp=datetime.utcnow(),
            )

            # 대화 기록에 응답 추가
            conversation.append(assistant_message)

            return ChatResponse(
                conversation_id=conversation_id,
                messages=conversation,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )

        except Exception as e:
            logger.error(f"Llama 응답 생성 실패: {str(e)}")
            return self._generate_test_response(conversation, conversation_id)

    def _generate_test_response(
        self, messages: List[Message], conversation_id: str
    ) -> ChatResponse:
        """테스트 모드에서 응답을 생성합니다."""
        logger.info("테스트 모드로 응답 생성")

        # 마지막 메시지의 내용을 기반으로 간단한 응답 생성
        last_message = messages[-1].content if messages else "Hello"

        # 간단한 규칙 기반 응답 생성
        if "날씨" in last_message:
            response = "오늘은 맑은 하늘이 펼쳐져 있고, 기온은 20도 정도로 쾌적한 날씨입니다. 산책하기 좋은 날이네요!"
        elif "이름" in last_message:
            response = "제 이름은 AI 어시스턴트입니다. 도움이 필요하신가요?"
        else:
            response = f"안녕하세요! '{last_message}'에 대한 응답입니다. 현재 테스트 모드로 실행 중입니다."

        # 새로운 메시지 생성
        assistant_message = Message(
            role=MessageRole.ASSISTANT, content=response, timestamp=datetime.utcnow()
        )

        # 대화 기록에 응답 추가
        messages.append(assistant_message)

        return ChatResponse(
            conversation_id=conversation_id,
            messages=messages,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
