# 이 파일은 이제 사용되지 않습니다.
# app.services.sllm_provider에서 제공하는 중앙 SLLM 서비스를 대신 사용하세요.

import os
import logging
import time
import uuid
from typing import List, Optional, Dict, AsyncGenerator
from datetime import datetime
from llama_cpp import Llama
from app.v3.models.chat import Message, ChatResponse, StreamingChatResponse, MessageRole
from app.config import get_settings
from app.provider.sllm_provider import get_llama_model, get_model_info

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SLLMService:
    """스트리밍 응답을 지원하는 채팅 서비스"""

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
        top_p: float = None,
        frequency_penalty: float = None,
        presence_penalty: float = None,
        rendered_prompt: Optional[str] = None,
    ) -> ChatResponse:
        """Llama 모델을 사용하여 응답을 생성하고 대화 기록을 관리합니다.

        Args:
            messages (List[Message]): 대화 메시지 목록
            conversation_id (Optional[str], optional): 대화 ID
            model (str, optional): 사용할 모델 이름
            temperature (float, optional): 생성 다양성 조절 파라미터
            max_tokens (Optional[int], optional): 최대 생성 토큰 수
            top_p (float, optional): 생성 다양성 조절 파라미터
            frequency_penalty (float, optional): 생성 다양성 조절 파라미터
            presence_penalty (float, optional): 생성 다양성 조절 파라미터
            rendered_prompt (Optional[str], optional): 프롬프트 서비스를 통해 렌더링된 프롬프트 문자열

        Returns:
            ChatResponse: 생성된 응답
        """
        logger.info(
            f"generate_response 호출됨: {messages}, {conversation_id}, {model}, {temperature}, {max_tokens}, {top_p}, {frequency_penalty}, {presence_penalty}"
        )

        # 기본값 설정
        model = model or self.default_model
        temperature = (
            temperature if temperature is not None else self.default_temperature
        )
        max_tokens = max_tokens or self.default_max_tokens
        top_p = top_p if top_p is not None else 1.0
        frequency_penalty = frequency_penalty if frequency_penalty is not None else 0.0
        presence_penalty = presence_penalty if presence_penalty is not None else 0.0
        conversation_id = conversation_id or str(uuid.uuid4())

        # 대화 기록 가져오기
        conversation = self._get_or_create_conversation(conversation_id)

        # 각 메시지에 타임스탬프 추가
        for msg in messages:
            if msg.timestamp is None:
                msg.timestamp = datetime.utcnow()

        # 메시지 복사본 생성
        messages_copy = []
        for msg in messages:
            # 모든 필드를 명시적으로 설정
            message_dict = {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp,
                "metadata": None  # 명시적으로 None으로 설정
            }
            # metadata 필드가 있는 경우에만 추가
            if hasattr(msg, "metadata") and msg.metadata is not None:
                message_dict["metadata"] = msg.metadata
            messages_copy.append(Message(**message_dict))

        # 복사본을 대화 기록에 추가
        conversation.extend(messages_copy)

        # 모델이 로드되지 않은 경우 테스트 응답 반환
        if not self.model_loaded:
            logger.warning("모델이 로드되지 않았습니다. 테스트 응답을 반환합니다.")
            return self._generate_test_response(messages_copy, conversation_id)

        try:
            # 프롬프트 구성 (렌더링된 프롬프트가 있으면 그것을 사용, 없으면 직접 구성)
            if rendered_prompt:
                prompt = rendered_prompt
                logger.info("렌더링된 프롬프트 사용")
            else:
                # 기존 방식으로 프롬프트 구성
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

            # Llama 모델로 응답 생성
            start_time = time.time()
            response = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=["Human:", "System:", "Assistant:"],
            )
            inference_time = time.time() - start_time
            logger.info(f"추론 시간: {inference_time:.2f}초")

            generated_text = response["choices"][0]["text"].strip()
            logger.info(f"모델 응답: {generated_text}")

            if not generated_text:
                generated_text = "죄송합니다. 응답을 생성할 수 없습니다. 다시 시도해주세요."

            # 응답 메시지 생성
            assistant_message = Message(
                role=MessageRole.ASSISTANT,
                content=generated_text,
                timestamp=datetime.utcnow(),
                metadata=None  # 명시적으로 None으로 설정
            )
            print(f"assistant_message: {assistant_message}")

            # 대화 기록에 응답 추가
            conversation.append(assistant_message)

            # 토큰 사용량 계산
            usage = {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(generated_text.split()),
                "total_tokens": len(prompt.split()) + len(generated_text.split()),
            }

            logger.info(f"사용량: {usage}")
            logger.info(f"모델: {model}")
            logger.info(f"inference_time: {inference_time}")
            logger.info(f"conversation: {conversation}")
            logger.info(f"conversation_id: {conversation_id}")
            logger.info(f"created_at: {datetime.utcnow()}")
            logger.info(f"updated_at: {datetime.utcnow()}")

            # ChatResponse 생성
            chat_response = ChatResponse(
                conversation_id=conversation_id,
                messages=conversation,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                usage=usage,
                metadata={"model": model, "inference_time": inference_time}
            )

            # ChatResponse 유효성 검사
            logger.info(f"ChatResponse: {chat_response}")
            logger.info(f"ChatResponse.messages: {chat_response.messages}")

            return chat_response

        except Exception as e:
            logger.error(f"Llama 응답 생성 실패: {str(e)}")
            logger.info("테스트 모드로 응답 생성")
            return self._generate_test_response(messages_copy, conversation_id)

    async def generate_streaming_response(
        self,
        messages: List[Message],
        conversation_id: Optional[str] = None,
        model: str = None,
        temperature: float = None,
        max_tokens: Optional[int] = None,
        top_p: float = None,
        frequency_penalty: float = None,
        presence_penalty: float = None,
        rendered_prompt: Optional[str] = None,
    ) -> AsyncGenerator[StreamingChatResponse, None]:
        """Llama 모델을 사용하여 스트리밍 응답을 생성합니다.

        Args:
            messages (List[Message]): 대화 메시지 목록
            conversation_id (Optional[str], optional): 대화 ID
            model (str, optional): 사용할 모델 이름
            temperature (float, optional): 생성 다양성 조절 파라미터
            max_tokens (Optional[int], optional): 최대 생성 토큰 수
            top_p (float, optional): 생성 다양성 조절 파라미터
            frequency_penalty (float, optional): 생성 다양성 조절 파라미터
            presence_penalty (float, optional): 생성 다양성 조절 파라미터
            rendered_prompt (Optional[str], optional): 프롬프트 서비스를 통해 렌더링된 프롬프트 문자열

        Yields:
            StreamingChatResponse: 생성된 응답의 일부분
        """
        logger.info(
            f"generate_streaming_response 호출됨: {messages}, {conversation_id}, {model}, {temperature}, {max_tokens}, {top_p}, {frequency_penalty}, {presence_penalty}"
        )

        # 기본값 설정
        model = model or self.default_model
        temperature = (
            temperature if temperature is not None else self.default_temperature
        )
        max_tokens = max_tokens or self.default_max_tokens
        top_p = top_p if top_p is not None else 1.0
        frequency_penalty = frequency_penalty if frequency_penalty is not None else 0.0
        presence_penalty = presence_penalty if presence_penalty is not None else 0.0
        conversation_id = conversation_id or str(uuid.uuid4())

        # 대화 기록 가져오기
        conversation = self._get_or_create_conversation(conversation_id)
        conversation.extend(messages)

        # 중앙 관리되는 모델이 로드되지 않은 경우 테스트 모드로 실행
        if not self.model_loaded or self.model is None:
            logger.warning("모델이 로드되지 않았습니다. 테스트 모드로 실행됩니다.")
            async for response in self._generate_test_streaming_response(
                conversation, conversation_id
            ):
                yield response
            return

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

            # 스트리밍 응답 생성
            response = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=["Human:", "System:", "Assistant:"],
                stream=True,
            )

            accumulated_text = ""
            accumulated_tokens = 0

            for chunk in response:
                if "choices" in chunk and len(chunk["choices"]) > 0:
                    text_chunk = chunk["choices"][0].get("text", "")
                    accumulated_text += text_chunk
                    accumulated_tokens += len(text_chunk.split())

                    # 토큰 사용량 계산
                    usage = {
                        "prompt_tokens": len(prompt.split()),
                        "completion_tokens": accumulated_tokens,
                        "total_tokens": len(prompt.split()) + accumulated_tokens,
                    }

                    # 스트리밍 응답 생성
                    yield StreamingChatResponse(
                        conversation_id=conversation_id,
                        content=text_chunk,
                        is_complete=False,
                        usage=usage,
                        metadata={"model": model},
                    )

            # 마지막 응답 생성
            inference_time = time.time() - start_time
            logger.info(f"스트리밍 추론 시간: {inference_time:.2f}초")
            logger.info(f"전체 모델 응답: {accumulated_text}")

            # 새로운 메시지 생성
            assistant_message = Message(
                role=MessageRole.ASSISTANT,
                content=accumulated_text,
                timestamp=datetime.utcnow(),
            )

            # 대화 기록에 응답 추가
            conversation.append(assistant_message)

            # 마지막 응답 전송
            yield StreamingChatResponse(
                conversation_id=conversation_id,
                content="",
                is_complete=True,
                usage=usage,
                metadata={"model": model, "inference_time": inference_time},
            )

        except Exception as e:
            logger.error(f"Llama 스트리밍 응답 생성 실패: {str(e)}")
            async for response in self._generate_test_streaming_response(
                conversation, conversation_id
            ):
                yield response

    async def _generate_test_streaming_response(
        self, messages: List[Message], conversation_id: str
    ) -> AsyncGenerator[StreamingChatResponse, None]:
        """테스트 모드에서 스트리밍 응답을 생성합니다."""
        logger.info("테스트 모드로 스트리밍 응답 생성")

        # 마지막 메시지의 내용을 기반으로 간단한 응답 생성
        last_message = messages[-1].content if messages else "Hello"

        # 간단한 규칙 기반 응답 생성
        if "날씨" in last_message:
            response_parts = [
                "오늘은 맑은 하늘이 펼쳐져 있고, ",
                "기온은 20도 정도로 쾌적한 날씨입니다. ",
                "산책하기 좋은 날이네요!",
            ]
        elif "이름" in last_message:
            response_parts = ["제 이름은 AI 어시스턴트입니다. ", "도움이 필요하신가요?"]
        else:
            response_parts = [
                f"안녕하세요! '{last_message}'에 대한 응답입니다. ",
                "현재 테스트 모드로 실행 중입니다.",
            ]

        accumulated_text = ""
        accumulated_tokens = 0

        # 각 부분을 스트리밍으로 반환
        for i, part in enumerate(response_parts):
            accumulated_text += part
            accumulated_tokens += len(part.split())

            # 토큰 사용량 계산
            usage = {
                "prompt_tokens": len(last_message.split()),
                "completion_tokens": accumulated_tokens,
                "total_tokens": len(last_message.split()) + accumulated_tokens,
            }

            yield StreamingChatResponse(
                conversation_id=conversation_id,
                content=part,
                is_complete=(i == len(response_parts) - 1),
                usage=usage,
                metadata={"model": "test"},
            )

        # 새로운 메시지 생성
        assistant_message = Message(
            role=MessageRole.ASSISTANT,
            content=accumulated_text,
            timestamp=datetime.utcnow(),
        )

        # 대화 기록에 응답 추가
        messages.append(assistant_message)

        # 마지막 응답 생성
        yield StreamingChatResponse(
            conversation_id=conversation_id,
            content="",
            is_complete=True,
            usage=usage,
            metadata={"model": "test", "mode": "test"},
        )

    def _generate_test_response(
        self, messages: List[Message], conversation_id: str
    ) -> ChatResponse:
        """테스트 모드에서 사용되는 응답 생성 메서드

        Args:
            messages (List[Message]): 대화 메시지 목록
            conversation_id (str): 대화 ID

        Returns:
            ChatResponse: 생성된 응답
        """
        # 메시지 복사본 생성
        messages_copy = []
        for msg in messages:
            # 타임스탬프가 없는 경우 현재 시간으로 설정
            timestamp = msg.timestamp if msg.timestamp is not None else datetime.utcnow()
            # 새 메시지 객체 생성
            message_dict = {
                "role": msg.role,
                "content": msg.content,
                "timestamp": timestamp
            }
            # metadata 필드가 있는 경우에만 추가
            if hasattr(msg, "metadata") and msg.metadata is not None:
                message_dict["metadata"] = msg.metadata
            messages_copy.append(Message(**message_dict))

        # 테스트 응답 메시지 생성
        assistant_message = Message(
            role=MessageRole.ASSISTANT,
            content="Hello, I am a helpful AI assistant. How can I assist you today?",
            timestamp=datetime.utcnow()
        )

        # 복사본에 응답 추가
        messages_copy.append(assistant_message)

        return ChatResponse(
            conversation_id=conversation_id,
            messages=messages_copy,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            metadata={"model": "test-model"}
        )
