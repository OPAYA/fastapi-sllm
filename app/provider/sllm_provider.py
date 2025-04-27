import logging
from typing import Dict, Any, Optional
import os
import time
from llama_cpp import Llama
from app.config import get_settings

logger = logging.getLogger(__name__)

class LlamaModelProvider:
    """
    중앙 Llama 모델 제공자 클래스
    모든 버전의 SLLM 서비스가 공유할 수 있는 모델 인스턴스를 관리합니다.
    싱글톤 패턴을 구현하되 global 변수를 사용하지 않습니다.
    """
    _instance: Optional['LlamaModelProvider'] = None

    def __new__(cls):
        if cls._instance is None:
            logger.info("LlamaModelProvider 인스턴스 생성")
            cls._instance = super(LlamaModelProvider, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not getattr(self, "_initialized", False):
            logger.info("LlamaModelProvider 초기화")
            settings = get_settings()
            self.model_path = settings.MODEL_PATH
            self.default_model = settings.DEFAULT_MODEL
            self.model = None
            self.model_loaded = False
            self.model_load_time = 0

            # 모델 로드 시도
            self._load_model()
            self._initialized = True

    def _load_model(self):
        """모델을 로드합니다."""
        if not os.path.exists(self.model_path):
            logger.warning(f"모델 파일을 찾을 수 없습니다: {self.model_path}")
            logger.warning("모델이 로드되지 않았습니다.")
            return

        try:
            logger.info(f"모델 로딩 중: {self.model_path}")
            start_time = time.time()
            self.model = Llama(
                model_path=self.model_path, n_ctx=2048, n_threads=8, n_gpu_layers=1
            )
            self.model_load_time = time.time() - start_time
            self.model_loaded = True
            logger.info(f"모델 로딩 완료! 소요 시간: {self.model_load_time:.2f}초")
        except Exception as e:
            logger.error(f"모델 로딩 실패: {str(e)}")
            logger.warning("모델이 로드되지 않았습니다.")

    @property
    def is_loaded(self) -> bool:
        """모델이 로드되었는지 확인합니다."""
        return self.model_loaded and self.model is not None

    def get_model(self) -> Llama:
        """
        로드된 Llama 모델 인스턴스를 반환합니다.
        모델이 로드되지 않은 경우 None을 반환합니다.
        """
        return self.model if self.is_loaded else None


# FastAPI 의존성 주입에서 사용할 함수
def get_llama_model() -> Optional[Llama]:
    """
    FastAPI 의존성 주입 시스템에서 사용할 함수입니다.
    Llama 모델 인스턴스를 반환합니다.
    """
    provider = LlamaModelProvider()
    return provider.get_model()

def get_model_info() -> dict:
    """
    모델 정보를 반환합니다.
    """
    provider = LlamaModelProvider()
    return {
        "model_loaded": provider.model_loaded,
        "model_load_time": provider.model_load_time,
        "model_path": provider.model_path,
        "default_model": provider.default_model
    }
