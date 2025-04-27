# FastAPI SLLM API

FastAPI 기반의 Small Language Model (SLLM) API 서버입니다. 이 프로젝트는 다양한 언어 모델을 통합하여 대화형 AI 서비스를 제공합니다.

## 기능

- GGUF 및 ONNX 모델 지원
- 대화 기록 관리 (메모리 기반)
- 스트리밍 응답 지원
- 프롬프트 템플릿 시스템
- 버전별 API 엔드포인트 (v1, v2, v3, v4)
- Swagger UI를 통한 API 문서화

## 설치 방법

### 요구사항

- Python 3.8 이상
- Poetry (패키지 관리자)

### 설치 단계

1. 저장소 클론
```bash
git clone https://github.com/yourusername/fastapi-sllm-api.git
cd fastapi-sllm-api
```

2. Poetry 설치 (없는 경우)
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3. 의존성 설치
```bash
poetry install
```

4. 환경 변수 설정
`.env` 파일을 생성하고 다음 내용을 추가합니다:
```
MODEL_PATH=/path/to/your/model
MAX_HISTORY_LENGTH=100
PAGE_SIZE=20
```

## 실행 방법

개발 서버 실행:
```bash
poetry run uvicorn app.main:app --reload
```

또는 `run.sh` 스크립트 사용:
```bash
./run.sh
```

## API 엔드포인트

### v4 API (최신)

#### 채팅
- `POST /v4/chat`: 일반 채팅 요청
- `POST /v4/chat/stream`: 스트리밍 채팅 요청
- `GET /v4/conversations`: 대화 목록 조회
- `GET /v4/conversations/{conversation_id}`: 특정 대화 조회
- `DELETE /v4/conversations/{conversation_id}`: 대화 삭제

### v3 API

#### 채팅
- `POST /v3/chat`: 일반 채팅 요청
- `POST /v3/chat/stream`: 스트리밍 채팅 요청

### v2 API

#### 채팅
- `POST /v2/chat`: 일반 채팅 요청
- `POST /v2/chat/stream`: 스트리밍 채팅 요청

### v1 API

#### 채팅
- `POST /v1/chat`: 일반 채팅 요청

## 프로젝트 구조

```
fastapi-sllm-api/
├── app/
│   ├── v1/                 # v1 API 구현
│   ├── v2/                 # v2 API 구현
│   ├── v3/                 # v3 API 구현
│   ├── v4/                 # v4 API 구현
│   │   ├── models/         # 데이터 모델
│   │   ├── routers/        # API 라우터
│   │   └── services/       # 비즈니스 로직
│   └── main.py             # 애플리케이션 진입점
├── models/                 # 모델 파일 저장
├── .env                    # 환경 변수
├── pyproject.toml          # Poetry 설정
└── README.md               # 프로젝트 문서
```



## 프롬프트 템플릿

v3 API에서는 다양한 프롬프트 템플릿을 지원합니다:

- `default.json`: 일반 대화용
- `code_assistant.json`: 프로그래밍 도우미
- `therapy_assistant.json`: 심리 상담
- `creative_writer.json`: 창작 도우미
- `sql_assistant.json`: SQL 도우미

## 라이선스

MIT License

## 기여 방법

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
