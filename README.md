## 개요
FastAPI SLLM API는 Small Language Model(SLLM)을 기반으로 한 대화형 AI 서비스를 제공하는 API 서버입니다. 다양한 언어 모델 형식(GGUF, ONNX)을 지원하며, 비전 관련 기능까지 확장된 종합적인 AI API 서비스를 제공합니다.

## 핵심 기능
- **다양한 모델 형식 지원**: GGUF 및 ONNX 모델 지원
- **멀티 버전 API**: v1부터 v5까지 단계적으로 발전된 API 제공
- **대화 관리**: 대화 기록 관리 및 컨텍스트 유지
- **스트리밍 응답**: 실시간 스트리밍 응답 지원
- **프롬프트 템플릿**: 사용 목적에 맞는 다양한 프롬프트 템플릿 지원
- **비전 AI**: 이미지 처리 및 분석 기능 (v5)
- **API 문서화**: Swagger UI를 통한 자동 API 문서화

## 시스템 요구사항
- Python 3.9 이상
- Poetry (패키지 관리)
- 최소 8GB RAM (모델 로딩용)
- CUDA 지원 GPU (선택사항, 성능 향상)

## 설치 방법

### 1. 저장소 클론
```bash
git clone https://github.com/yourusername/fastapi-sllm-api.git
cd fastapi-sllm-api
```

### 2. Poetry 설치
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### 3. 의존성 설치
```bash
poetry install
```

### 기본 실행
```bash
./run.sh
```
또는
```bash
poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```


## 프로젝트 구조

```
fastapi-sllm-api/
├── app/                           # 메인 애플리케이션 디렉토리
│   ├── main.py                    # 애플리케이션 진입점
│   ├── config.py                  # 전역 설정 파일
│   │
│   ├── provider/                  # 모델 제공자 관련 코드
│   │   ├── model_provider.py      # 기본 모델 제공자 인터페이스
│   │   ├── sllm_provider.py       # SLLM 모델 제공자 구현
│   │   ├── vision_provider.py     # 비전 모델 제공자 구현
│   │   ├── gguf_provider.py       # GGUF 모델 처리
│   │   └── onnx_provider.py       # ONNX 모델 처리
│   │
│   ├── v1/                        # v1 API 구현 (기본 채팅)
│   │   ├── models/                # 데이터 모델
│   │   │   └── chat.py           # 채팅 관련 모델 정의
│   │   ├── routers/               # API 라우터
│   │   │   └── chat.py           # 채팅 관련 엔드포인트
│   │   └── services/              # 비즈니스 로직
│   │       └── chat_service.py    # 채팅 서비스 구현
│   │
│   ├── v2/                        # v2 API 구현 (스트리밍 채팅 추가)
│   │   ├── models/                # 데이터 모델
│   │   ├── routers/               # API 라우터
│   │   │   └── chat.py           # 스트리밍 지원 채팅 엔드포인트
│   │   └── services/              # 비즈니스 로직
│   │       └── stream_service.py  # 스트리밍 서비스 구현
│   │
│   ├── v3/                        # v3 API 구현 (프롬프트 템플릿 추가)
│   │   ├── models/                # 데이터 모델
│   │   ├── routers/               # API 라우터
│   │   │   ├── chat.py           # 채팅 엔드포인트
│   │   │   └── prompt.py         # 프롬프트 관리 엔드포인트
│   │   ├── services/              # 비즈니스 로직
│   │   └── prompts/               # 프롬프트 템플릿 정의
│   │
│   ├── v4/                        # v4 API 구현 (대화 관리 추가)
│   │   ├── config.py              # v4 전용 설정
│   │   ├── dependencies.py        # 의존성 주입 설정
│   │   ├── models/                # 데이터 모델
│   │   │   ├── chat.py           # 채팅 모델
│   │   │   ├── prompt.py         # 프롬프트 모델
│   │   │   └── schemas.py        # 공통 스키마
│   │   ├── routers/               # API 라우터
│   │   │   ├── chat.py           # 채팅 및 대화 관리 엔드포인트
│   │   │   └── prompt.py         # 프롬프트 관리 엔드포인트
│   │   ├── services/              # 비즈니스 로직
│   │   │   ├── chat_service.py   # 채팅 서비스
│   │   │   ├── chat_history.py   # 대화 기록 관리
│   │   │   ├── sllm_service.py   # SLLM 모델 서비스
│   │   │   ├── prompt_service.py # 프롬프트 서비스
│   │   │   └── conversation_service.py # 대화 관리 서비스
│   │   └── prompts/               # 프롬프트 템플릿
│   │
│   └── v5/                        # v5 API 구현 (비전 기능 추가)
│       ├── config.py              # v5 전용 설정
│       ├── dependencies.py        # 의존성 주입 설정
│       ├── models/                # 데이터 모델
│       │   └── vision.py         # 비전 관련 모델 정의
│       ├── routers/               # API 라우터
│       │   └── vision.py         # 비전 관련 엔드포인트
│       └── services/              # 비즈니스 로직
│           └── vision_service.py  # 비전 서비스 구현
│
├── models/                        # 모델 파일 저장 디렉토리
│   ├── gguf/                     # GGUF 모델 파일
│   ├── onnx/                     # ONNX 모델 파일
│   └── vision/                   # 비전 모델 파일
│
├── api_docs/                      # API 문서 디렉토리
│   └── ...                       # 정적 API 문서 파일들
│
├── export_static_docs.py          # API 문서 내보내기 스크립트
├── export_docs.py                 # API 문서 생성 스크립트
├── export_docs_fixed.py           # 수정된 API 문서 생성 스크립트
├── test.py                        # 테스트 스크립트
├── .env                          # 환경 변수 설정 파일
├── pyproject.toml                # Poetry 프로젝트 설정
├── poetry.lock                   # Poetry 의존성 잠금 파일
├── run.sh                        # 서버 실행 스크립트
└── README.md                     # 프로젝트 문서
```

## API 엔드포인트

### v1 API (기본 채팅)
- `POST /api/v1/chat`: 기본 채팅 요청

### v2 API (스트리밍 추가)
- `POST /api/v2/chat`: 일반 채팅 요청
- `POST /api/v2/chat/stream`: 스트리밍 채팅 요청

### v3 API (프롬프트 템플릿 추가)
- `POST /api/v3/chat`: 일반 채팅 요청
- `POST /api/v3/chat/stream`: 스트리밍 채팅 요청
- `GET /api/v3/prompt/templates`: 프롬프트 템플릿 목록
- `GET /api/v3/prompt/templates/{name}`: 특정 템플릿 조회

### v4 API (대화 관리 추가)
- `POST /api/v4/chat`: 일반 채팅 요청
- `POST /api/v4/chat/stream`: 스트리밍 채팅 요청
- `GET /api/v4/conversations`: 대화 목록 조회
- `GET /api/v4/conversations/{id}`: 특정 대화 조회
- `DELETE /api/v4/conversations/{id}`: 대화 삭제
- `GET /api/v4/prompt/templates`: 프롬프트 템플릿 목록
- `POST /api/v4/prompt/templates`: 새 템플릿 생성
- `PUT /api/v4/prompt/templates/{name}`: 템플릿 업데이트
- `DELETE /api/v4/prompt/templates/{name}`: 템플릿 삭제

### v5 API (비전 기능 추가)
- `POST /api/v5/vision/chat`: 이미지 분석 채팅
- `GET /api/v5/vision/info`: 비전 모델 정보 조회

### 공통 엔드포인트
- `GET /`: API 소개 정보
- `GET /health`: 서버 및 모델 상태 확인

## 지원 프롬프트 템플릿

v3 API부터는 다양한 용도의 프롬프트 템플릿을 지원합니다:

- **default.json**: 일반 대화용 템플릿
- **code_assistant.json**: 프로그래밍 도우미 템플릿
- **therapy_assistant.json**: 심리 상담용 템플릿
- **creative_writer.json**: 창작 도우미 템플릿
- **sql_assistant.json**: SQL 도우미 템플릿

## HyperCLOVAX 모델 다운로드

HyperCLOVAX Vision 모델을 다운로드하려면 다음 Python 코드를 실행하세요:

```python
from huggingface_hub import login, snapshot_download
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 환경 변수에서 토큰 가져오기
token = os.getenv("HUGGINGFACE_TOKEN")
if not token:
    raise ValueError("HUGGINGFACE_TOKEN 환경 변수가 설정되지 않았습니다. .env 파일에 토큰을 설정해주세요.")

# Hugging Face 토큰으로 로그인
login(token=token)

# 모델 다운로드 경로 설정
download_path = "./hyperclovax_vision_3b"

# 디렉토리가 없으면 생성
if not os.path.exists(download_path):
    os.makedirs(download_path)

print("HyperCLOVAX Vision 모델 다운로드를 시작합니다...")

# 모델 다운로드
snapshot_download(
    repo_id="naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B",
    local_dir=download_path,
    local_dir_use_symlinks=False
)

print(f"모델 다운로드가 완료되었습니다. 저장 위치: {download_path}")
```

이 코드를 실행하기 전에 `.env` 파일에 Hugging Face 토큰을 설정해야 합니다:

```bash
echo "HUGGINGFACE_TOKEN=your_token_here" >> .env
```

## 모델 설치

### 모델 다운로드
1. [Google Drive 링크](https://drive.google.com/drive/folders/1kVT6YZZeRZGPWpPE9ZNzaPTbGkQdacqi?usp=sharing)에서 다음 모델 파일들을 다운로드하세요:
   - `onnx_qwen25.onnx`
   - `phi-2-q4_k_m.gguf`

2. 프로젝트 루트 디렉토리에 `models` 폴더를 생성하고 다운로드한 모델 파일들을 이 폴더에 넣으세요:
```bash
mkdir models
# 다운로드한 모델 파일들을 models/ 폴더로 이동
```
