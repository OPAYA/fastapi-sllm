from typing import List, Dict, Optional
from datetime import datetime
from ..models.schemas import Message as MessageSchema

class ChatHistory:
    def __init__(self):
        self.conversations: Dict[str, Dict] = {}
        self.messages: Dict[str, List[MessageSchema]] = {}

    def add_message(self, conversation_id: str, role: str, content: str, metadata: Optional[Dict] = None) -> MessageSchema:
        if conversation_id not in self.messages:
            self.messages[conversation_id] = []

        message = MessageSchema(
            role=role,
            content=content,
            timestamp=datetime.utcnow(),
            metadata=metadata or {}
        )
        self.messages[conversation_id].append(message)
        return message

    def get_conversation(self, conversation_id: str) -> Optional[Dict]:
        return self.conversations.get(conversation_id)

    def get_messages(self, conversation_id: str, limit: Optional[int] = None) -> List[MessageSchema]:
        messages = self.messages.get(conversation_id, [])
        if limit:
            return messages[-limit:]
        return messages

    def create_conversation(self, title: str = "New Conversation", metadata: Optional[Dict] = None) -> Dict:
        conversation_id = str(len(self.conversations) + 1)
        conversation = {
            "id": conversation_id,
            "title": title,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "metadata": metadata or {}
        }
        self.conversations[conversation_id] = conversation
        self.messages[conversation_id] = []
        return conversation

    def update_conversation(self, conversation_id: str, title: Optional[str] = None, metadata: Optional[Dict] = None) -> Optional[Dict]:
        conversation = self.get_conversation(conversation_id)
        if conversation:
            if title:
                conversation["title"] = title
            if metadata:
                conversation["metadata"].update(metadata)
            conversation["updated_at"] = datetime.utcnow()
        return conversation

    def delete_conversation(self, conversation_id: str) -> bool:
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            if conversation_id in self.messages:
                del self.messages[conversation_id]
            return True
        return False

    def get_all_conversations(self, skip: int = 0, limit: int = 20) -> List[Dict]:
        conversations = list(self.conversations.values())
        conversations.sort(key=lambda x: x["updated_at"], reverse=True)
        return conversations[skip:skip + limit]

    def get_conversation_summary(self, conversation_id: str) -> Dict:
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return {}

        messages = self.get_messages(conversation_id)
        return {
            "id": conversation["id"],
            "title": conversation["title"],
            "created_at": conversation["created_at"],
            "updated_at": conversation["updated_at"],
            "message_count": len(messages),
            "metadata": conversation["metadata"]
        }

# 전역 인스턴스 생성
chat_history = ChatHistory()
