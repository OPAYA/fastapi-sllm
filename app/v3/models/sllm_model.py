import logging
from typing import AsyncGenerator
from ..services.sllm_service import SLLMService

logger = logging.getLogger(__name__)

class SLLMModel:
    def __init__(self):
        self.service = SLLMService()
        logger.info("SLLMModel initialized")

    async def generate(self, message: str) -> str:
        """
        메시지를 처리하고 응답을 생성합니다.
        """
        try:
            response = await self.service.generate_response(message)
            return response
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise

    async def generate_stream(self, message: str) -> AsyncGenerator[str, None]:
        """
        스트리밍 방식으로 메시지를 처리하고 응답을 생성합니다.
        """
        try:
            async for chunk in self.service.generate_response_stream(message):
                yield f"data: {chunk}\n\n"
        except Exception as e:
            logger.error(f"Error generating streaming response: {str(e)}")
            yield f'data: {{"error": "{str(e)}"}}\n\n'
