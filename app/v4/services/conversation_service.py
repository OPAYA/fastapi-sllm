from typing import List
from app.v3.models.chat import Message
import logging

logger = logging.getLogger(__name__)


class ConversationService:
    def __init__(self):
        # 데이터베이스 연결 등 초기화 로직
        pass

    async def save_conversation(self, conversation_id: str, messages: List[Message]):
        """대화 기록을 저장합니다."""
        try:
            # TODO: 실제 데이터베이스에 대화 저장 로직 구현
            logger.info(
                f"Saved conversation {conversation_id} with {len(messages)} messages"
            )
        except Exception as e:
            logger.error(f"Error saving conversation {conversation_id}: {str(e)}")

    async def get_conversation(self, conversation_id: str) -> List[Message]:
        """대화 기록을 불러옵니다."""
        try:
            # TODO: 실제 데이터베이스에서 대화 불러오기 로직 구현
            logger.info(f"Retrieved conversation {conversation_id}")
            return []  # 임시 빈 배열 반환
        except Exception as e:
            logger.error(f"Error retrieving conversation {conversation_id}: {str(e)}")
            return []
