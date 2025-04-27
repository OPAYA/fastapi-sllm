import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import re
from app.v3.models.prompt import PromptTemplate, PromptResponse
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class PromptService:
    """프롬프트 관리 서비스"""

    def __init__(self, prompt_dir: Optional[str] = None):
        """
        프롬프트 서비스 초기화

        Args:
            prompt_dir: 프롬프트 템플릿 파일이 저장된 디렉토리 경로 (지정하지 않으면 기본 경로 사용)
        """
        self.prompt_dir = prompt_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../prompts")
        self.templates: Dict[str, PromptTemplate] = {}
        self._load_templates()

    def _load_templates(self) -> None:
        """템플릿 디렉토리에서 모든 프롬프트 템플릿을 로드합니다."""
        logger.info(f"프롬프트 템플릿 로드 중... 경로: {self.prompt_dir}")

        # 디렉토리가 없으면 생성
        os.makedirs(self.prompt_dir, exist_ok=True)

        try:
            # 기본 프롬프트 템플릿이 없으면 생성
            if not os.listdir(self.prompt_dir):
                self._create_default_templates()

            # 모든 템플릿 파일 로드
            for filename in os.listdir(self.prompt_dir):
                if filename.endswith(".json"):
                    file_path = os.path.join(self.prompt_dir, filename)
                    try:
                        with open(file_path, "r", encoding="utf-8") as file:
                            template_data = json.load(file)
                            template = PromptTemplate(**template_data)
                            self.templates[template.id] = template
                            logger.info(f"템플릿 로드됨: {template.id} ({template.name})")
                    except Exception as e:
                        logger.error(f"템플릿 로드 실패: {file_path}, 오류: {str(e)}")
        except Exception as e:
            logger.error(f"템플릿 디렉토리 로드 실패: {str(e)}")
            # 기본 템플릿 메모리에 로드
            self._create_default_templates(in_memory_only=True)

    def _create_default_templates(self, in_memory_only: bool = False) -> None:
        """기본 프롬프트 템플릿을 생성합니다."""
        default_templates = [
            PromptTemplate(
                id="default_chat",
                name="기본 채팅 프롬프트",
                description="일반적인 대화를 위한 기본 프롬프트 템플릿",
                template="You are a helpful AI assistant. Please respond directly and concisely to the user's questions.\n\n{{messages}}",
                version="1.0",
                tags=["chat", "general"],
                metadata={"author": "SSAFY Team", "created_at": datetime.now().isoformat()}
            ),
            PromptTemplate(
                id="coding_assistant",
                name="코딩 어시스턴트",
                description="코드 작성 및 디버깅을 돕는 프롬프트",
                template="You are a coding assistant who helps with programming tasks. You provide clear, concise, and correct code examples.\n\n{{messages}}",
                version="1.0",
                tags=["coding", "programming"],
                metadata={"author": "SSAFY Team", "created_at": datetime.now().isoformat()}
            )
        ]

        # 메모리에 로드
        for template in default_templates:
            self.templates[template.id] = template

            # 파일로 저장 (요청된 경우에만)
            if not in_memory_only:
                self.save_template(template)

    def get_template(self, template_id: str) -> Optional[PromptTemplate]:
        """템플릿 ID로 프롬프트 템플릿을 조회합니다."""
        return self.templates.get(template_id)

    def get_all_templates(self) -> List[PromptTemplate]:
        """모든 프롬프트 템플릿을 조회합니다."""
        return list(self.templates.values())

    def save_template(self, template: PromptTemplate) -> bool:
        """프롬프트 템플릿을 저장합니다."""
        try:
            # 메모리에 저장
            self.templates[template.id] = template

            # 파일로 저장
            file_path = os.path.join(self.prompt_dir, f"{template.id}.json")
            with open(file_path, "w", encoding="utf-8") as file:
                # Pydantic 모델을 JSON으로 직렬화
                file.write(template.json(ensure_ascii=False, indent=2))

            logger.info(f"템플릿 저장됨: {template.id} ({template.name})")
            return True
        except Exception as e:
            logger.error(f"템플릿 저장 실패: {template.id}, 오류: {str(e)}")
            return False

    def delete_template(self, template_id: str) -> bool:
        """프롬프트 템플릿을 삭제합니다."""
        if template_id not in self.templates:
            logger.warning(f"템플릿을 찾을 수 없음: {template_id}")
            return False

        try:
            # 메모리에서 삭제
            del self.templates[template_id]

            # 파일 삭제
            file_path = os.path.join(self.prompt_dir, f"{template_id}.json")
            if os.path.exists(file_path):
                os.remove(file_path)

            logger.info(f"템플릿 삭제됨: {template_id}")
            return True
        except Exception as e:
            logger.error(f"템플릿 삭제 실패: {template_id}, 오류: {str(e)}")
            return False

    def render_prompt(self, template_id: str, variables: Dict[str, Any] = None) -> PromptResponse:
        """
        프롬프트 템플릿을 렌더링합니다.

        Args:
            template_id: 렌더링할 템플릿 ID
            variables: 템플릿 변수

        Returns:
            PromptResponse: 렌더링된 프롬프트 응답

        Raises:
            ValueError: 템플릿을 찾을 수 없는 경우
        """
        variables = variables or {}
        template = self.get_template(template_id)

        if not template:
            raise ValueError(f"템플릿을 찾을 수 없음: {template_id}")

        # 템플릿 렌더링
        rendered_prompt = template.template

        # 변수 대체 (간단한 {{variable}} 구문 사용)
        for var_name, var_value in variables.items():
            placeholder = f"{{{{{var_name}}}}}"
            rendered_prompt = rendered_prompt.replace(placeholder, str(var_value))

        # 남은 변수 자리표시자 확인
        remaining_vars = re.findall(r'{{([^{}]+)}}', rendered_prompt)
        if remaining_vars:
            logger.warning(f"렌더링되지 않은 변수가 있습니다: {remaining_vars}")

        return PromptResponse(
            rendered_prompt=rendered_prompt,
            template_id=template_id,
            variables=variables,
            metadata={
                "rendered_at": datetime.now().isoformat(),
                "template_version": template.version
            }
        )
