from pydantic import BaseModel
from typing import Dict, Optional
from datetime import datetime

class Message(BaseModel):
    role: str
    content: str
    timestamp: datetime
    metadata: Dict = {}

class ChatRequest(BaseModel):
    conversation_id: Optional[str] = None
    message: str
    metadata: Optional[Dict] = None

class ChatResponse(BaseModel):
    conversation_id: str
    message: Message
    metadata: Dict = {}
