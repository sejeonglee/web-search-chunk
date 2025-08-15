#!/bin/bash

# 환경 변수 로드
if [ -f .env ]; then
    export $(cat .env | xargs)
fi

# 가상환경 활성화
source .venv/bin/activate

# 메인 실행
echo "Starting WebSearch QA System..."
python src/main.py "$@"

