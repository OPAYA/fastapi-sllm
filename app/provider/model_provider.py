from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

class ModelProvider(ABC):
    """모델 제공자 인터페이스"""

    @abstractmethod
    def load_model(self) -> bool:
        """모델을 로드합니다."""
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        """모델이 로드되었는지 확인합니다."""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보를 반환합니다."""
        pass

    @abstractmethod
    def run_inference(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """모델을 사용하여 추론을 실행합니다."""
        pass

    @abstractmethod
    def close(self) -> None:
        """모델 리소스를 해제합니다."""
        pass


