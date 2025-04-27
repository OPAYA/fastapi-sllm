import logging
from typing import Dict, Any, Optional
import os
import time
from llama_cpp import Llama
from app.config import get_settings
from app.provider.model_provider import ModelProvider

logger = logging.getLogger(__name__)

class GGUFModelProvider(ModelProvider):
    """
    GGUF 모델 제공자 클래스
    모든 버전의 SLLM 서비스가 공유할 수 있는 GGUF 모델 인스턴스를 관리합니다.
    싱글톤 패턴을 구현하되 global 변수를 사용하지 않습니다.
    """
    _instance: Optional['GGUFModelProvider'] = None

    def __new__(cls):
        if cls._instance is None:
            logger.info("GGUFModelProvider 인스턴스 생성")
            cls._instance = super(GGUFModelProvider, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not getattr(self, "_initialized", False):
            logger.info("GGUFModelProvider 초기화")
            settings = get_settings()
            self.model_path = settings.MODEL_PATH
            self.default_model = settings.DEFAULT_MODEL
            self.model = None
            self.model_loaded = False
            self.model_load_time = 0

            # 모델 로드 시도
            self.load_model()
            self._initialized = True

    def load_model(self) -> bool:
        """GGUF 모델을 로드합니다."""
        if not os.path.exists(self.model_path):
            logger.warning(f"모델 파일을 찾을 수 없습니다: {self.model_path}")
            logger.warning("모델이 로드되지 않았습니다.")
            return False

        try:
            logger.info(f"모델 로딩 중: {self.model_path}")
            start_time = time.time()
            self.model = Llama(
                model_path=self.model_path, n_ctx=2048, n_threads=8, n_gpu_layers=1
            )
            self.model_load_time = time.time() - start_time
            self.model_loaded = True
            logger.info(f"모델 로딩 완료! 소요 시간: {self.model_load_time:.2f}초")
            return True
        except Exception as e:
            logger.error(f"모델 로딩 실패: {str(e)}")
            logger.warning("모델이 로드되지 않았습니다.")
            return False

    def is_loaded(self) -> bool:
        """모델이 로드되었는지 확인합니다."""
        return self.model_loaded and self.model is not None

    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보를 반환합니다."""
        return {
            "model_loaded": self.model_loaded,
            "model_load_time": self.model_load_time,
            "model_path": self.model_path,
            "default_model": self.default_model
        }

    def run_inference(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        GGUF 모델을 사용하여 추론을 실행합니다.

        Args:
            input_data (Dict[str, Any]): 모델 입력 데이터

        Returns:
            Dict[str, Any]: 모델 출력 데이터
        """
        if not self.is_loaded():
            logger.error("모델이 로드되지 않았습니다.")
            return None

        try:
            # GGUF 모델의 입력 형식에 맞게 처리
            prompt = input_data.get("prompt", "")
            temperature = input_data.get("temperature", 0.7)
            max_tokens = input_data.get("max_tokens", 1000)

            # 모델 실행
            result = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["\n"]
            )

            return {
                "text": result["choices"][0]["text"].strip(),
                "usage": result["usage"]
            }
        except Exception as e:
            logger.error(f"추론 실행 중 오류 발생: {str(e)}")
            return None

    def close(self) -> None:
        """모델 리소스를 해제합니다."""
        if self.model is not None:
            # GGUF 모델은 명시적인 close 메서드가 없으므로 참조를 제거
            self.model = None
            self.model_loaded = False

def get_gguf_model() -> Llama:
    """
    GGUF 모델을 가져옵니다.
    """
    provider = GGUFModelProvider()
    return provider.model if provider.is_loaded() else None

def get_gguf_model_info() -> dict:
    """
    GGUF 모델 정보를 반환합니다.
    """
    provider = GGUFModelProvider()
    return provider.get_model_info()
