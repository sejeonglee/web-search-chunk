# WebSearch QA System

웹 검색 기반 질문-답변 시스템 (포트 & 어댑터 아키텍처)

## 🏗️ 아키텍처

### 의존성 역전 원칙 (DIP) 적용
- **Core**: 도메인 로직과 인터페이스(포트) 정의
- **Adapters**: 외부 서비스 연동을 위한 구체적인 구현체
- **Main**: 의존성 주입 및 조립

```
src/
├── core/                     # 도메인 계층 (인터페이스만 사용)
│   ├── models.py            # 도메인 모델 + 포트 인터페이스 정의
│   └── pipeline.py          # 비즈니스 로직 (인터페이스에만 의존)
│
├── adapters/                 # 어댑터 계층 (구체적 구현)
│   ├── llm_adapter.py       # ILLMService 구현
│   ├── web_search_adapter.py # IWebSearchService 구현
│   ├── crawling_adapter.py  # ICrawlingService 구현
│   ├── vector_store_adapter.py # IVectorStore 구현
│   ├── persistent_store_adapter.py # IPersistentStore 구현
│   ├── chunking_adapter.py  # IChunkingService 구현
│   ├── retrieval_adapter.py # IRetrievalService 구현
│   └── reranking_adapter.py # IRerankingService 구현
│
└── main.py                   # 의존성 주입 컨테이너 & 조립
```

### 핵심 인터페이스 (Ports)
```python
# core/models.py에 정의된 포트들
- ILLMService: LLM 서비스 인터페이스
- IWebSearchService: 웹 검색 인터페이스
- ICrawlingService: 크롤링 인터페이스
- IVectorStore: 벡터 저장소 인터페이스
- IPersistentStore: 영구 저장소 인터페이스
- IChunkingService: 청킹 서비스 인터페이스
- IRetrievalService: 검색 서비스 인터페이스
- IRerankingService: 리랭킹 서비스 인터페이스
```

### 구현체 선택 (Adapters)
```python
# 설정에 따라 다른 구현체 선택 가능
config = {
    "search_provider": "tavily",  # or "google"
    "chunking_strategy": "simple", # or "contextual"
}
```

## 🚀 설치 및 실행

### 설치
```bash
# uv 사용
uv venv
uv pip install -e .

# playwright 브라우저 설치
playwright install chromium
```

### 환경 변수 설정
```bash
# .env 파일 생성
TAVILY_API_KEY=your_api_key
GOOGLE_API_KEY=your_google_key  # optional
GOOGLE_CX=your_google_cx        # optional
LLM_MODEL=meta-llama/Llama-2-7b-chat-hf
SEARCH_PROVIDER=tavily          # or google
QDRANT_PATH=./qdrant_db
```

### 실행
```bash
# 메인 실행
python src/main.py

# 테스트 실행
pytest tests/ -v

# 린팅 & 타입 체크
ruff check src/ tests/
mypy src/ --ignore-missing-imports
```

## 🔧 주요 기능

### 1. 멀티 쿼리 재작성
- 사용자 질의를 다양한 검색 쿼리로 변환
- 한국어/영어 병렬 검색

### 2. 병렬/비동기 처리
- 웹 검색: 여러 쿼리 동시 처리
- 크롤링: 최대 10개 URL 병렬 크롤링
- 10초 이내 응답 보장

### 3. 하이브리드 검색
- BM25 + Vector 검색
- Reciprocal Rank Fusion (RRF)

### 4. 봇 방지
- User-Agent 랜덤화
- 요청 간 랜덤 지연
- 헤더 정보 위조

### 5. 세션 관리
- 단기: FAISS 인메모리
- 장기: Qdrant 영구 저장

## 🧪 테스트

### 단위 테스트
```bash
pytest tests/test_models.py -v
pytest tests/test_adapters.py -v
```

### 통합 테스트
```bash
pytest tests/test_pipeline.py -v
pytest tests/test_integration.py -v
```

### Mock 테스트
```bash
# 외부 의존성 없이 테스트
pytest tests/test_dependency_injection.py -v
```

## 📝 사용 예제

```python
from src.main import WebSearchQASystem

# 시스템 초기화
config = {
    "search_provider": "tavily",
    "chunking_strategy": "simple",
    "max_processing_time": 10.0
}
system = WebSearchQASystem(config)

# 쿼리 처리
result = await system.process_query("최근 AI 기술 동향은?")

if result["success"]:
    print(result["response"]["answer"])
    print(f"소스: {result['response']['sources']}")
```

## 🔄 확장 가능한 설계

### 새로운 검색 엔진 추가
```python
# src/adapters/new_search_adapter.py
class NewSearchAdapter(IWebSearchService):
    async def search(self, query: str, max_results: int = 7):
        # 구현
        pass

# main.py의 DependencyContainer에서 선택
if config["search_provider"] == "new_provider":
    return NewSearchAdapter()
```

### 새로운 LLM 추가
```python
# src/adapters/new_llm_adapter.py
class OpenAIAdapter(ILLMService):
    async def generate_queries(self, user_query: str):
        # OpenAI API 호출
        pass
```

## 📊 성능 목표

- **최대 실행 시간**: 10초 이내
- **최대 URL 크롤링**: 10개
- **청크 크기**: 1000자 (200자 오버랩)
- **최종 컨텍스트**: 상위 5개 청크

## 🐳 Docker 지원

```bash
# 빌드
docker-compose build

# 실행
docker-compose up
```

## 📄 라이선스

MIT License