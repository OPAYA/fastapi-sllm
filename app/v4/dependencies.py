import logging
from app.v3.services.chat_service import ChatService as V3ChatService
from app.v3.services.conversation_service import ConversationService
from app.v3.services.sllm_service import SLLMService
from app.v3.services.prompt_service import PromptService
from app.v4.services.chat_service import ChatService

logger = logging.getLogger(__name__)

# 싱글톤 패턴으로 서비스 인스턴스 관리
_sllm_service = None
_conversation_service = None
_prompt_service = None
_chat_service = None

def get_sllm_service():
    """
    SLLMService 인스턴스를 반환합니다.
    각 버전별 특화된 서비스 구현을 사용하되, 내부적으로는 공유된 모델 인스턴스를 사용합니다.
    """
    global _sllm_service
    if _sllm_service is None:
        logger.info("v3 SLLMService 인스턴스 생성")
        _sllm_service = SLLMService()
    return _sllm_service

def get_chat_service():
    """
    v4 ChatService 인스턴스를 반환합니다.
    """
    global _chat_service
    if _chat_service is None:
        logger.info("v4 ChatService 인스턴스 생성")
        _chat_service = ChatService(sllm_service=get_sllm_service())
    return _chat_service

def get_v3_chat_service():
    """
    v3 ChatService 인스턴스를 반환합니다.
    """
    # SLLMService 인스턴스를 주입
    return V3ChatService(sllm_service=get_sllm_service())

def get_conversation_service():
    """ConversationService 인스턴스를 반환합니다."""
    global _conversation_service
    if _conversation_service is None:
        logger.info("ConversationService 인스턴스 생성")
        _conversation_service = ConversationService()
    return _conversation_service

def get_prompt_service():
    """PromptService 인스턴스를 반환합니다."""
    global _prompt_service
    if _prompt_service is None:
        logger.info("PromptService 인스턴스 생성")
        _prompt_service = PromptService()
    return _prompt_service
