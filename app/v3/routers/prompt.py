from fastapi import APIRouter, HTTPException, Depends, Path, Query
from typing import List, Dict, Any, Optional
from app.v3.models.prompt import PromptTemplate, PromptRequest, PromptResponse
from app.v3.services.prompt_service import PromptService
from app.v3.dependencies import get_prompt_service
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/templates", response_model=List[PromptTemplate])
async def get_all_templates(
    prompt_service: PromptService = Depends(get_prompt_service),
    tag: Optional[str] = Query(None, description="필터링할 태그"),
):
    """
    모든 프롬프트 템플릿을 조회합니다.

    - **tag**: 특정 태그로 필터링 (선택 사항)
    """
    templates = prompt_service.get_all_templates()

    # 태그로 필터링
    if tag:
        templates = [t for t in templates if tag in t.tags]

    return templates


@router.get("/templates/{template_id}", response_model=PromptTemplate)
async def get_template(
    template_id: str = Path(..., description="조회할 템플릿 ID"),
    prompt_service: PromptService = Depends(get_prompt_service),
):
    """
    특정 ID의 프롬프트 템플릿을 조회합니다.

    - **template_id**: 조회할 템플릿 ID
    """
    template = prompt_service.get_template(template_id)
    if not template:
        raise HTTPException(status_code=404, detail=f"템플릿을 찾을 수 없음: {template_id}")
    return template


@router.post("/templates", response_model=PromptTemplate)
async def create_template(
    template: PromptTemplate,
    prompt_service: PromptService = Depends(get_prompt_service),
):
    """
    새 프롬프트 템플릿을 생성합니다.

    - **template**: 새로 생성할 템플릿 정보
    """
    # 이미 존재하는 템플릿 ID인지 확인
    if prompt_service.get_template(template.id):
        raise HTTPException(status_code=400, detail=f"이미 존재하는 템플릿 ID: {template.id}")

    success = prompt_service.save_template(template)
    if not success:
        raise HTTPException(status_code=500, detail="템플릿 저장 실패")

    return template


@router.put("/templates/{template_id}", response_model=PromptTemplate)
async def update_template(
    template: PromptTemplate,
    template_id: str = Path(..., description="업데이트할 템플릿 ID"),
    prompt_service: PromptService = Depends(get_prompt_service),
):
    """
    기존 프롬프트 템플릿을 업데이트합니다.

    - **template_id**: 업데이트할 템플릿 ID
    - **template**: 업데이트할 템플릿 정보
    """
    # 템플릿 존재 확인
    if not prompt_service.get_template(template_id):
        raise HTTPException(status_code=404, detail=f"템플릿을 찾을 수 없음: {template_id}")

    # ID 불일치 확인
    if template.id != template_id:
        raise HTTPException(status_code=400, detail="URL의 템플릿 ID와 요청 본문의 템플릿 ID가 일치하지 않습니다")

    success = prompt_service.save_template(template)
    if not success:
        raise HTTPException(status_code=500, detail="템플릿 업데이트 실패")

    return template


@router.delete("/templates/{template_id}", response_model=Dict[str, Any])
async def delete_template(
    template_id: str = Path(..., description="삭제할 템플릿 ID"),
    prompt_service: PromptService = Depends(get_prompt_service),
):
    """
    프롬프트 템플릿을 삭제합니다.

    - **template_id**: 삭제할 템플릿 ID
    """
    # 템플릿 존재 확인
    if not prompt_service.get_template(template_id):
        raise HTTPException(status_code=404, detail=f"템플릿을 찾을 수 없음: {template_id}")

    success = prompt_service.delete_template(template_id)
    if not success:
        raise HTTPException(status_code=500, detail="템플릿 삭제 실패")

    return {"status": "success", "message": f"템플릿 삭제됨: {template_id}"}


@router.post("/render", response_model=PromptResponse)
async def render_prompt(
    request: PromptRequest,
    prompt_service: PromptService = Depends(get_prompt_service),
):
    """
    프롬프트 템플릿을 렌더링합니다.

    - **request**: 프롬프트 렌더링 요청
    """
    try:
        return prompt_service.render_prompt(
            template_id=request.template_id,
            variables=request.variables
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"프롬프트 렌더링 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"프롬프트 렌더링 실패: {str(e)}")
