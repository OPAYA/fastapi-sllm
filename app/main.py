from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import logging
from app.v1.routers import chat as chat_v1
from app.v2.routers import chat as chat_v2
from app.v3.routers import chat as chat_v3
from app.v3.routers import prompt as prompt_v3
from app.v4.routers import chat as chat_v4
from app.v4.routers import prompt as prompt_v4
from app.v5.routers import vision as vision_v5
from app.v1.models.chat import Message, MessageRole
from app.config import get_settings
from app.provider.sllm_provider import LlamaModelProvider, get_model_info
from app.provider.vision_provider import VisionModelProvider, get_vision_model_info

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="SLLM API",
    description="Small Language Model API for conversational AI services",
    version="1.0.0",
    docs_url="/docs",
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(chat_v1.router, prefix="/api/v1", tags=["chat-v1"])
app.include_router(chat_v2.router, prefix="/api/v2", tags=["chat-v2"])
app.include_router(chat_v3.router, prefix="/api/v3", tags=["chat-v3"])
app.include_router(prompt_v3.router, prefix="/api/v3/prompt", tags=["prompt-v3"])
app.include_router(chat_v4.router, prefix="/api/v4", tags=["chat-v4"])
app.include_router(prompt_v4.router, prefix="/api/v4/prompt", tags=["prompt-v4"])
app.include_router(vision_v5.router, prefix="/api/v5", tags=["vision-v5"])


@app.on_event("startup")
async def startup_event():
    """서버 시작 시 실행되는 이벤트 핸들러"""
    logger.info("서버 시작 중...")

    try:
        # SLLM 모델 로드 - 싱글톤 인스턴스를 생성하여 메모리에 로드합니다
        sllm_provider = LlamaModelProvider()
        sllm_info = get_model_info()

        if sllm_info["model_loaded"]:
            logger.info(f"SLLM 모델 로드 성공! 로드 시간: {sllm_info['model_load_time']:.2f}초")
        else:
            logger.warning("SLLM 모델 로드 실패! 테스트 모드로 실행됩니다.")

        # Vision 모델 로드
        vision_provider = VisionModelProvider()
        vision_info = get_vision_model_info()

        if vision_info["model_loaded"]:
            logger.info(f"Vision 모델 로드 성공! 로드 시간: {vision_info['model_load_time']:.2f}초")
        else:
            logger.warning("Vision 모델 로드 실패! 테스트 모드로 실행됩니다.")
    except Exception as e:
        logger.error(f"서버 시작 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=f"서버 시작 실패: {str(e)}")


@app.get("/")
async def root():
    """루트 경로 핸들러"""
    return {
        "message": "SLLM API에 오신 것을 환영합니다!",
        "version": "1.0.0",
        "documentation": "/docs",
    }


@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    sllm_info = get_model_info()
    vision_info = get_vision_model_info()

    return {
        "status": "healthy",
        "sllm_model": {
            "loaded": sllm_info["model_loaded"],
            "load_time": sllm_info["model_load_time"],
        },
        "vision_model": {
            "loaded": vision_info["model_loaded"],
            "load_time": vision_info["model_load_time"],
        }
    }
