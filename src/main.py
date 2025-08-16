"""Main entry point - Dependency Injection and Assembly."""

import asyncio
import os
import uuid
from typing import Optional
from datetime import datetime

# Core domain (인터페이스만 사용)
from .core.pipeline import QAPipeline
from .core.models import IPersistentStore

# Concrete implementations (어댑터)
from .adapters.llm_adapter import VLLMAdapter
from .adapters.web_search_adapter import TavilySearchAdapter, GoogleSearchAdapter
from .adapters.crawling_adapter import PlaywrightCrawler
from .adapters.vector_store_adapter import FAISSVectorStore
from .adapters.persistent_store_adapter import QdrantPersistentStore
from .adapters.chunking_adapter import SimpleChunkingAdapter, ContextualChunkingAdapter
from .adapters.retrieval_adapter import HybridRetrievalAdapter
from .adapters.reranking_adapter import CrossEncoderRerankingAdapter


class DependencyContainer:
    """의존성 컨테이너 - 구체적인 구현체를 생성하고 주입."""

    def __init__(self, config: dict, session_id: str):
        self.config = config
        self.session_id = session_id
        self._instances = {}

    def get_llm_service(self):
        """LLM 서비스 인스턴스 반환."""
        if "llm_service" not in self._instances:
            self._instances["llm_service"] = VLLMAdapter(
                model_name=self.config["llm_model"],
                embedding_model=self.config.get("embedding_model", "bge-large:335m"),
                base_url=self.config.get("vllm_base_url", "http://localhost:8000/v1")
            )
        return self._instances["llm_service"]

    def get_search_service(self):
        """검색 서비스 인스턴스 반환."""
        if "search_service" not in self._instances:
            # 설정에 따라 다른 구현체 선택
            if self.config.get("search_provider") == "google":
                self._instances["search_service"] = GoogleSearchAdapter(
                    api_key=self.config["google_api_key"], cx=self.config["google_cx"]
                )
            else:
                self._instances["search_service"] = TavilySearchAdapter(
                    api_key=self.config["tavily_api_key"]
                )
        return self._instances["search_service"]

    def get_crawling_service(self):
        """크롤링 서비스 인스턴스 반환."""
        if "crawling_service" not in self._instances:
            self._instances["crawling_service"] = PlaywrightCrawler()
        return self._instances["crawling_service"]

    def get_vector_store(self):
        """벡터 저장소 인스턴스 반환."""
        if "vector_store" not in self._instances:
            self._instances["vector_store"] = FAISSVectorStore(
                dimension=self.config["vector_dimension"]
            )
        return self._instances["vector_store"]

    def get_persistent_store(self) -> IPersistentStore:
        """영구 저장소 인스턴스 반환."""
        if "persistent_store" not in self._instances:
            self._instances["persistent_store"] = QdrantPersistentStore(
                session_id=self.session_id,
                host=self.config.get("qdrant_host", "localhost"),
                port=self.config.get("qdrant_port", 6333)
            )
        return self._instances["persistent_store"]

    def get_chunking_service(self):
        """청킹 서비스 인스턴스 반환."""
        if "chunking_service" not in self._instances:
            if self.config.get("chunking_strategy") == "contextual":
                self._instances["chunking_service"] = ContextualChunkingAdapter(
                    llm_service=self.get_llm_service(),
                    chunk_size=self.config["chunk_size"],
                    overlap=self.config["chunk_overlap"],
                )
            else:
                self._instances["chunking_service"] = SimpleChunkingAdapter(
                    chunk_size=self.config["chunk_size"],
                    overlap=self.config["chunk_overlap"],
                )
        return self._instances["chunking_service"]

    def get_retrieval_service(self):
        """검색 서비스 인스턴스 반환."""
        if "retrieval_service" not in self._instances:
            self._instances["retrieval_service"] = HybridRetrievalAdapter(
                vector_store=self.get_vector_store(),
                llm_service=self.get_llm_service()
            )
        return self._instances["retrieval_service"]

    def get_reranking_service(self):
        """리랭킹 서비스 인스턴스 반환."""
        if "reranking_service" not in self._instances:
            self._instances["reranking_service"] = CrossEncoderRerankingAdapter()
        return self._instances["reranking_service"]

    def build_pipeline(self) -> QAPipeline:
        """파이프라인 조립 - 모든 의존성 주입."""
        return QAPipeline(
            llm_service=self.get_llm_service(),
            search_service=self.get_search_service(),
            crawling_service=self.get_crawling_service(),
            vector_store=self.get_vector_store(),
            chunking_service=self.get_chunking_service(),
            retrieval_service=self.get_retrieval_service(),
            reranking_service=self.get_reranking_service(),
        )


class WebSearchQASystem:
    """웹 검색 QA 시스템 - 최상위 조립."""

    def __init__(self, config: Optional[dict] = None, session_id: Optional[str] = None):
        self.config = config or self._get_default_config()
        self.session_id = session_id or str(uuid.uuid4())
        self.container = DependencyContainer(self.config, self.session_id)
        self.pipeline = self.container.build_pipeline()
        self.persistent_store = self.container.get_persistent_store()

    def _get_default_config(self) -> dict:
        """기본 설정."""
        return {
            "llm_model": "Qwen/Qwen3-4B-Instruct-2507-FP8",
            "embedding_model": "bge-large:335m",
            "vllm_base_url": "http://localhost:8000/v1",
            "search_provider": "tavily",  # "tavily" or "google"
            "tavily_api_key": os.getenv("TAVILY_API_KEY", ""),
            "google_api_key": os.getenv("GOOGLE_API_KEY", ""),
            "google_cx": os.getenv("GOOGLE_CX", ""),
            "vector_dimension": 1024,  # bge-large:335m은 1024차원
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "chunking_strategy": "contextual",  # "simple" or "contextual"
            "max_processing_time": 10.0,
            "qdrant_host": "localhost",
            "qdrant_port": 6333,
        }

    async def process_query(self, query: str) -> dict:
        """쿼리 처리 (10초 타임아웃)."""
        try:
            # 이전 세션 데이터 로드
            await self._load_session_data()

            # 타임아웃 설정
            response = await asyncio.wait_for(
                self.pipeline.run(query), timeout=self.config["max_processing_time"]
            )

            # 세션 데이터 저장
            await self._save_session_data()

            return {
                "success": True,
                "response": response.model_dump(),
                "processing_time": response.processing_time,
            }

        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": "Processing exceeded 10 seconds timeout",
                "partial_response": None,
            }
        except Exception as e:
            print(f"❌ 처리 중 오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e), "partial_response": None}

    async def _load_session_data(self):
        """세션 데이터 로드."""
        try:
            chunks = await self.persistent_store.load_session()
            if chunks:
                await self.container.get_vector_store().add_chunks(chunks)
                print(f"🔄 세션 '{self.session_id}'에서 {len(chunks)}개 기존 청크를 로드했습니다.")
        except:
            pass  # 첫 실행 시 무시

    async def _save_session_data(self):
        """세션 데이터 저장."""
        try:
            vector_store = self.container.get_vector_store()
            # FAISSVectorStore의 chunks 속성 접근
            if hasattr(vector_store, "chunks"):
                print(f"💾 Vector store에 있는 chunks 개수: {len(vector_store.chunks)}")
                if vector_store.chunks:
                    await self.persistent_store.save_session(vector_store.chunks)
                    print(f"✅ 세션 '{self.session_id}'에 {len(vector_store.chunks)}개 chunk를 저장했습니다.")
                else:
                    print("⚠️  Vector store에 저장된 chunks가 없습니다.")
            else:
                print("❌ Vector store에 chunks 속성이 없습니다.")
        except Exception as e:
            print(f"Failed to save session: {e}")
            import traceback
            traceback.print_exc()


async def main():
    """메인 실행 함수."""
    # 설정 로드 (환경 변수 또는 설정 파일에서)
    config = {
        "llm_model": os.getenv("LLM_MODEL", "Qwen/Qwen3-4B-Instruct-2507-FP8"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "bge-large:335m"),
        "vllm_base_url": os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
        "search_provider": os.getenv("SEARCH_PROVIDER", "tavily"),
        "tavily_api_key": os.getenv("TAVILY_API_KEY", ""),
        "vector_dimension": 1024,  # bge-large:335m은 1024차원
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "chunking_strategy": "contextual",
        "max_processing_time": 10.0,
        "qdrant_host": os.getenv("QDRANT_HOST", "localhost"),
        "qdrant_port": int(os.getenv("QDRANT_PORT", "6333")),
    }

    # 시스템 초기화
    system = WebSearchQASystem(config)
    print(f"🔑 세션 ID: {system.session_id}")

    # 예제 쿼리 처리
    queries = ["최근 AI 기술 동향은?", "2024년 한국 경제 전망", "기후 변화 대응 방안"]

    for query in queries:
        print(f"\n처리 중: {query}")
        print("-" * 50)

        start_time = datetime.now()
        result = await system.process_query(query)

        if result["success"]:
            response = result["response"]
            print(f"답변: {response['answer'][:200]}...")
            print(f"소스: {', '.join(response['sources'][:3])}")
            print(f"처리 시간: {result['processing_time']:.2f}초")
        else:
            print(f"오류: {result['error']}")


if __name__ == "__main__":
    # 이벤트 루프 실행
    asyncio.run(main())
