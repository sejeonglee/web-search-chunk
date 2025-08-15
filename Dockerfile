FROM python:3.11-slim

WORKDIR /app

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# uv 설치
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# 프로젝트 파일 복사
COPY pyproject.toml .
COPY src/ src/
COPY tests/ tests/

# 의존성 설치
RUN uv pip install --system -e .

# Playwright 설치
RUN playwright install --with-deps chromium

CMD ["python", "src/main.py"]