import os
import logging
import time
import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime
from llama_cpp import Llama
from app.v1.models.chat import Message, ChatResponse, MessageRole
from app.config import get_settings
from app.provider.gguf_provider import GGUFModelProvider
from app.provider.onnx_provider import OnnxModelProvider

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SLLMService:
    """
    SLLM 서비스 클래스
    GGUF와 ONNX 모델을 모두 지원하는 통합 서비스
    """

    def __init__(self):
        settings = get_settings()
        self.default_model = settings.DEFAULT_MODEL
        self.default_temperature = 0.7
        self.default_max_tokens = 1000

        # 모델 제공자 초기화
        self.gguf_provider = GGUFModelProvider()
        self.onnx_provider = OnnxModelProvider()

    @property
    def model_loaded(self) -> bool:
        """모델이 로드되었는지 확인합니다."""
        return self.gguf_provider.is_loaded() or self.onnx_provider.is_loaded()

    @property
    def model_load_time(self) -> float:
        """모델 로드 시간을 반환합니다."""
        return max(
            self.gguf_provider.get_model_info().get("model_load_time", 0),
            self.onnx_provider.get_model_info().get("model_load_time", 0)
        )

    def _get_provider(self, model_name: str) -> Optional[Any]:
        """
        모델 이름에 따라 적절한 제공자를 반환합니다.

        Args:
            model_name (str): 모델 이름

        Returns:
            Optional[Any]: 모델 제공자 또는 None
        """
        if model_name.startswith("onnx_"):
            return self.onnx_provider
        return self.gguf_provider

    async def generate_response(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> ChatResponse:
        """
        주어진 메시지에 대한 응답을 생성합니다.

        Args:
            messages (List[Message]): 대화 메시지 목록
            model (Optional[str]): 사용할 모델 이름
            temperature (Optional[float]): 생성 온도
            max_tokens (Optional[int]): 최대 토큰 수

        Returns:
            ChatResponse: 생성된 응답
        """
        # 기본값 설정
        model = model or self.default_model
        temperature = temperature or self.default_temperature
        max_tokens = max_tokens or self.default_max_tokens

        logger.info(f"모델 {model}로 응답 생성 시작 (temperature={temperature}, max_tokens={max_tokens})")

        # 적절한 제공자 선택
        provider = self._get_provider(model)
        if not provider or not provider.is_loaded():
            logger.warning(f"모델 {model}이(가) 로드되지 않았습니다.")
            return self._generate_test_response(messages)

        try:
            # 프롬프트 구성
            prompt = self._construct_prompt(messages)

            # 입력 데이터 준비
            input_data = {
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens
            }

            # 모델 실행
            start_time = time.time()
            result = provider.run_inference(input_data)
            generation_time = time.time() - start_time

            if result is None:
                logger.error("모델 실행 결과가 None입니다.")
                return self._generate_test_response(messages)

            # 응답 구성
            response = ChatResponse(
                model=model,
                choices=[{
                    "message": Message(
                        role="assistant",
                        content=result["text"]
                    ),
                    "finish_reason": "stop"
                }],
                usage=result["usage"],
                generation_time=generation_time
            )

            logger.info(f"응답 생성 완료 (소요 시간: {generation_time:.2f}초)")
            return response

        except Exception as e:
            logger.error(f"응답 생성 중 오류 발생: {str(e)}")
            return self._generate_test_response(messages)

    def _construct_prompt(self, messages: List[Message]) -> str:
        """
        메시지 목록으로부터 프롬프트를 구성합니다.

        Args:
            messages (List[Message]): 대화 메시지 목록

        Returns:
            str: 구성된 프롬프트
        """
        prompt_parts = []
        for msg in messages:
            role = msg.role.upper()
            content = msg.content
            prompt_parts.append(f"{role}: {content}")
        return "\n".join(prompt_parts)

    def _generate_test_response(self, messages: List[Message]) -> ChatResponse:
        """
        테스트 응답을 생성합니다.

        Args:
            messages (List[Message]): 대화 메시지 목록

        Returns:
            ChatResponse: 테스트 응답
        """
        return ChatResponse(
            model="test",
            choices=[{
                "message": Message(
                    role="assistant",
                    content="죄송합니다. 모델이 로드되지 않았거나 오류가 발생했습니다."
                ),
                "finish_reason": "error"
            }],
            usage={
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            },
            generation_time=0.0
        )

    def close(self) -> None:
        """모델 리소스를 해제합니다."""
        self.gguf_provider.close()
        self.onnx_provider.close()

    async def get_available_models(self) -> List[str]:
        """사용 가능한 모델 목록을 반환합니다."""
        # 실제 구현에서는 사용 가능한 모델 목록을 반환하도록 구현합니다.
        # 현재는 간단히 기본 모델만 반환합니다.
        return [self.default_model]
