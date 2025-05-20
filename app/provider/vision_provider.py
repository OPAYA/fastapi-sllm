import logging
import time
from typing import Optional, Dict, Any, List
import os
from transformers import AutoTokenizer, AutoProcessor, AutoConfig, AutoModelForCausalLM
import torch
from PIL import Image
import base64
from io import BytesIO
from .model_provider import ModelProvider

# hyperclovax import
from hyperclovax_vision_3b.modeling_hyperclovax import HCXVisionForCausalLM, HCXVisionConfig

logger = logging.getLogger(__name__)

class VisionModelProvider(ModelProvider):
    """HyperCloVAX Vision 모델 제공자"""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.device = "cpu"
        self.model_loaded = False

    def load_model(self) -> bool:
        """모델을 로드합니다."""
        try:
            logger.info(f"Loading vision model from {self.model_path}")

            # 커스텀 등록
            AutoConfig.register("hyperclovax_vlm", HCXVisionConfig)
            AutoModelForCausalLM.register(HCXVisionConfig, HCXVisionForCausalLM)

            # 모델 로드
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True
            ).to(self.device)

            self.model_loaded = True
            logger.info("Vision model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load vision model: {str(e)}")
            self.model_loaded = False
            return False

    def is_loaded(self) -> bool:
        """모델이 로드되었는지 확인합니다."""
        return self.model_loaded

    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보를 반환합니다."""
        return {
            "model_path": self.model_path,
            "device": self.device,
            "is_loaded": self.model_loaded
        }

    def run_inference(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """모델을 사용하여 추론을 실행합니다."""
        if not self.model_loaded:
            raise RuntimeError("Model is not loaded")

        try:
            messages = input_data.get("messages", [])
            if not messages:
                raise ValueError("No messages provided")

            # 이미지와 비디오 처리
            new_messages, all_images, is_video_list = self.processor.load_images_videos(messages)
            preprocessed = self.processor(all_images, is_video_list=is_video_list)

            # 토큰화
            inputs = self.tokenizer.apply_chat_template(
                new_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.device)

            # 추론
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=8192,
                    do_sample=True,
                    top_p=0.6,
                    temperature=0.5,
                    repetition_penalty=1.0,
                    **preprocessed
                )

            # 결과 디코딩
            response = self.tokenizer.batch_decode(outputs)[0]

            return {
                "response": response,
                "status": "success"
            }

        except Exception as e:
            logger.error(f"Error during inference: {str(e)}")
            return {
                "error": str(e),
                "status": "error"
            }

    def close(self) -> None:
        """모델 리소스를 해제합니다."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        self.model_loaded = False
        torch.cuda.empty_cache()

# FastAPI 의존성 주입에서 사용할 함수
def get_vision_tokenizer():
    """토크나이저 인스턴스를 반환합니다."""
    provider = VisionModelProvider("./hyperclovax-vision-3B")
    if not provider.is_loaded():
        provider.load_model()
    return provider.tokenizer

def get_vision_processor():
    """프로세서 인스턴스를 반환합니다."""
    provider = VisionModelProvider("./hyperclovax-vision-3B")
    if not provider.is_loaded():
        provider.load_model()
    return provider.processor

def get_vision_model():
    """모델 인스턴스를 반환합니다."""
    provider = VisionModelProvider("./hyperclovax-vision-3B")
    if not provider.is_loaded():
        provider.load_model()
    return provider.model

def get_vision_model_info() -> Dict[str, Any]:
    """모델 정보를 반환합니다."""
    provider = VisionModelProvider("./hyperclovax-vision-3B")
    if not provider.is_loaded():
        provider.load_model()
    return provider.get_model_info()
