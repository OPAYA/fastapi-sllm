import logging
from app.v2.services.chat_service import ChatService
from app.v2.services.sllm_service import SLLMService

logger = logging.getLogger(__name__)

# 싱글톤 패턴으로 서비스 인스턴스 관리
_sllm_service = None

def get_sllm_service():
    """
    SLLMService 인스턴스를 반환합니다.
    각 버전별 특화된 서비스 구현을 사용하되, 내부적으로는 공유된 모델 인스턴스를 사용합니다.
    """
    global _sllm_service
    if _sllm_service is None:
        logger.info("v2 SLLMService 인스턴스 생성")
        _sllm_service = SLLMService()
    return _sllm_service

def get_chat_service():
    """
    ChatService 인스턴스를 반환합니다.
    """
    # SLLMService 인스턴스를 주입
    return ChatService(sllm_service=get_sllm_service())
