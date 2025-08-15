#!/bin/bash
# setup.sh - 프로젝트 설정 스크립트

echo "WebSearch QA System Setup"
echo "========================="

# Python 버전 확인
python_version=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
required_version="3.11"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then 
    echo "Error: Python $required_version or higher is required"
    exit 1
fi

# uv 설치 확인
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# 가상환경 생성
echo "Creating virtual environment..."
uv venv

# 의존성 설치
echo "Installing dependencies..."
uv pip install -e .

# Playwright 브라우저 설치
echo "Installing Playwright browsers..."
playwright install chromium

# 디렉터리 생성
mkdir -p qdrant_db
mkdir -p logs

echo "Setup complete!"
echo "Run 'source .venv/bin/activate' to activate the environment"

