#!/bin/bash

# 가상환경 활성화
source .venv/bin/activate

echo "Running tests..."

# 린팅
echo "Running linter..."
ruff check src/ tests/

# 타입 체크
echo "Running type checker..."
mypy src/ --ignore-missing-imports

# 테스트 실행
echo "Running pytest..."
pytest tests/ -v --cov=src --cov-report=term-missing

echo "Tests complete!"

