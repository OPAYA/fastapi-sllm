import logging
import time
from typing import List, Dict, Any, Optional, Union
import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM
from app.v5.models.vision import (
    VisionChatMessage,
    VisionChatRequest,
    VisionChatResponse,
)
from app.v5.config import DEFAULT_VISION_CONFIG

logger = logging.getLogger(__name__)


class VisionService:
    """HyperCloVAX Vision 모델 서비스"""

    def __init__(
        self,
        tokenizer,
        processor,
        model,
    ):
        """
        VisionService 생성자

        Args:
            tokenizer: HyperCloVAX Vision 토크나이저
            processor: HyperCloVAX Vision 프로세서
            model: HyperCloVAX Vision 모델
        """
        self.tokenizer = tokenizer
        self.processor = processor
        self.model = model
        self.config = DEFAULT_VISION_CONFIG
        self.device = self.config.device
        logger.info("VisionService 초기화 완료")

    @property
    def is_loaded(self) -> bool:
        """모델이 로드되었는지 확인합니다."""
        return (
            self.tokenizer is not None
            and self.processor is not None
            and self.model is not None
        )

    def _handle_test_mode(self, messages: List[VisionChatMessage]) -> str:
        """테스트 모드에서 응답을 생성합니다."""
        logger.warning("테스트 모드로 실행됩니다.")

        # 마지막 메시지 분석
        last_message = messages[-1]
        if last_message.role != "user":
            return "대화 흐름이 잘못되었습니다. 마지막 메시지는 사용자 메시지여야 합니다."

        # 테스트용 응답 생성
        content = last_message.content
        if isinstance(content, list):
            # 여러 콘텐츠가 있는 경우
            has_image = any(c.type == "image" for c in content if hasattr(c, "type"))
            has_video = any(c.type == "video" for c in content if hasattr(c, "type"))

            if has_video:
                return "이 비디오에는 사람들이 활동하는 모습이 보입니다. 자세한 설명을 원하시면 알려주세요."
            elif has_image:
                return "이 이미지에는 여러 객체가 보입니다. 무엇에 대해 더 알고 싶으신가요?"
            else:
                return "안녕하세요! 어떻게 도와드릴까요?"
        else:
            # 단일 콘텐츠
            if hasattr(content, "type"):
                if content.type == "image":
                    return (
                        f"이 이미지({content.filename})에 대해 설명해 드리겠습니다. 이미지에는 여러 객체가 보입니다."
                    )
                elif content.type == "video":
                    return f"이 비디오({content.filename})에 대해 설명해 드리겠습니다. 비디오에는 여러 활동이 포함되어 있습니다."
                else:
                    return f"안녕하세요! '{content.text}'에 대한 답변입니다. 어떻게 도와드릴까요?"
            else:
                return "안녕하세요! 어떻게 도와드릴까요?"

    async def generate_response(self, request: VisionChatRequest) -> VisionChatResponse:
        """
        HyperCloVAX Vision 모델을 사용하여 응답을 생성합니다.

        Args:
            request: 비전 채팅 요청

        Returns:
            비전 채팅 응답
        """
        start_time = time.time()

        # 모델이 로드되지 않은 경우 테스트 모드로 실행
        if not self.is_loaded:
            test_response = self._handle_test_mode(request.messages)
            inference_time = time.time() - start_time

            return VisionChatResponse(
                response=test_response,
                usage={
                    "prompt_tokens": 0,
                    "completion_tokens": len(test_response.split()),
                    "total_tokens": len(test_response.split()),
                },
                metadata={
                    "model": "test_mode",
                    "inference_time": inference_time,
                },
            )

        try:
            # 요청 파라미터 설정
            temperature = request.temperature or self.config.temperature
            top_p = request.top_p or self.config.top_p
            max_new_tokens = request.max_new_tokens or self.config.max_new_tokens
            repetition_penalty = (
                request.repetition_penalty or self.config.repetition_penalty
            )

            # 메시지 처리
            messages = request.messages

            # 이미지 및 비디오 처리
            new_vlm_chat, all_images, is_video_list = self.processor.load_images_videos(
                messages
            )
            preprocessed = self.processor(all_images, is_video_list=is_video_list)

            # 채팅 템플릿 적용
            input_ids = self.tokenizer.apply_chat_template(
                new_vlm_chat,
                return_tensors="pt",
                tokenize=True,
                add_generation_prompt=True,
            )

            # 모델 생성
            output_ids = self.model.generate(
                input_ids=input_ids.to(device=self.device),
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                **preprocessed,
            )

            # 응답 디코딩
            generated_text = self.tokenizer.batch_decode(output_ids)[0]

            # 생성된 응답 추출 (모델 출력에서 응답 부분만 추출)
            response_text = self._extract_response(generated_text)

            # 전체 처리 시간 계산
            inference_time = time.time() - start_time

            # 토큰 사용량 추정
            input_token_count = len(input_ids[0])
            output_token_count = len(output_ids[0]) - len(input_ids[0])

            return VisionChatResponse(
                response=response_text,
                usage={
                    "prompt_tokens": input_token_count,
                    "completion_tokens": output_token_count,
                    "total_tokens": input_token_count + output_token_count,
                },
                metadata={
                    "model": "hyperclovax-vision-3B",
                    "inference_time": inference_time,
                    "parameters": {
                        "temperature": temperature,
                        "top_p": top_p,
                        "max_new_tokens": max_new_tokens,
                        "repetition_penalty": repetition_penalty,
                    },
                },
            )

        except Exception as e:
            logger.error(f"응답 생성 중 오류 발생: {str(e)}")
            error_response = f"죄송합니다. 응답 생성 중 오류가 발생했습니다: {str(e)}"

            return VisionChatResponse(
                response=error_response,
                usage={
                    "prompt_tokens": 0,
                    "completion_tokens": len(error_response.split()),
                    "total_tokens": len(error_response.split()),
                },
                metadata={
                    "model": "hyperclovax-vision-3B",
                    "error": str(e),
                    "inference_time": time.time() - start_time,
                },
            )

    def _extract_response(self, generated_text: str) -> str:
        """생성된 텍스트에서 모델 응답만 추출합니다."""
        # HyperCloVAX 모델에 맞게 응답 추출 로직 구현
        # 이 부분은 모델의 출력 형식에 따라 조정이 필요할 수 있습니다

        # 생성된 텍스트에서 마지막 assistant 응답 부분만 추출
        try:
            # 마지막 assistant 응답 추출
            if "<|assistant|>" in generated_text:
                response = generated_text.split("<|assistant|>")[-1].strip()
                # 다음 토큰이 있으면 제거
                if "<|" in response:
                    response = response.split("<|")[0].strip()
                return response
            else:
                # 기본 동작: 전체 생성 텍스트 반환
                return generated_text.strip()
        except Exception as e:
            logger.error(f"응답 추출 중 오류 발생: {str(e)}")
            return generated_text.strip()
