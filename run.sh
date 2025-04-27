#!/bin/bash

# 색상 정의
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting FastAPI LLM API Server...${NC}"

# 현재 디렉토리가 fastapi-llm-api인지 확인
if [ ! -f "pyproject.toml" ]; then
    echo "Error: Please run this script from the fastapi-llm-api directory"
    exit 1
fi

# Poetry가 설치되어 있는지 확인
if ! command -v poetry &> /dev/null; then
    echo "Poetry is not installed. Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
fi

# 의존성 설치 확인
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}Installing dependencies...${NC}"
    poetry install
fi

# 모델 파일 확인
if [ ! -f "models/phi-2-q4_k_m.gguf" ]; then
    echo -e "${YELLOW}Warning: Model file not found. Please make sure the model file exists in the models directory.${NC}"
    echo -e "${YELLOW}The first request might be slow due to model loading.${NC}"
fi

# 가상 환경 활성화 및 서버 실행
echo -e "${GREEN}Starting server...${NC}"
echo -e "${BLUE}Note: The first request might be slow as the model is being loaded into memory.${NC}"
echo -e "${BLUE}Subsequent requests will be faster.${NC}"
poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
