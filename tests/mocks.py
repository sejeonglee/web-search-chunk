"""Mock implementations for testing without external dependencies."""

from typing import List, Optional, Dict, Any
from datetime import datetime
import random
import hashlib

from src.core.models import (
    SearchQuery,
    WebDocument,
    WebDocumentContent,
    SemanticChunk,
    QAResponse,
)
from src.adapters.llm_client import LLMClient
from src.adapters.web_search_adapter import WebSearchAdapter
from src.adapters.crawling_adapter import CrawlingAdapter
from src.adapters.vector_db_adapter import VectorDBAdapter


class MockLLMClient(LLMClient):
    """Mock LLM 클라이언트."""

    async def generate_queries(self, user_query: str) -> List[SearchQuery]:
        """Mock 쿼리 생성."""
        # 간단한 쿼리 변형 생성
        queries = [
            SearchQuery(
                original_query=user_query,
                processed_queries=[
                    user_query,
                    f"{user_query} 최신",
                    f"{user_query} 2024",
                ],
            ),
            SearchQuery(
                original_query=user_query,
                processed_queries=[f"what is {user_query}", f"{user_query} explained"],
                language="en",
            ),
        ]
        return queries

    async def generate_answer(self, query: str, context: str) -> str:
        """Mock 답변 생성."""
        return f"Based on the provided context about '{query}', here is a comprehensive answer. The context mentions several key points that are relevant to your question. [Mock answer generated for testing purposes]"


class MockWebSearchAdapter(WebSearchAdapter):
    """Mock 웹 검색 어댑터."""

    def __init__(self):
        self.mock_results = [
            {
                "url": "https://example.com/article1",
                "title": "Understanding AI and Machine Learning",
                "snippet": "AI and ML are transforming industries...",
            },
            {
                "url": "https://example.com/article2",
                "title": "Latest Technology Trends 2024",
                "snippet": "The year 2024 brings exciting developments...",
            },
            {
                "url": "https://example.com/article3",
                "title": "Data Science Best Practices",
                "snippet": "Effective data science requires...",
            },
        ]

    async def search(self, query: str, max_results: int = 7) -> List[WebDocument]:
        """Mock 검색 결과 반환."""
        results = []
        for i, item in enumerate(self.mock_results[:max_results]):
            results.append(
                WebDocument(
                    url=item["url"],
                    title=item["title"],
                    snippet=item["snippet"],
                    search_query=query,
                )
            )
        return results


class MockCrawlingAdapter(CrawlingAdapter):
    """Mock 크롤링 어댑터."""

    def __init__(self):
        self.mock_contents = {
            "https://example.com/article1": """
# Understanding AI and Machine Learning

Artificial Intelligence (AI) and Machine Learning (ML) are revolutionizing 
how we interact with technology. These technologies enable computers to 
learn from data and make intelligent decisions.

## Key Concepts

Machine learning algorithms can be categorized into supervised, unsupervised, 
and reinforcement learning. Each approach has its unique applications and 
benefits.

## Applications

From healthcare to finance, AI/ML applications are everywhere. They help in 
predictive analytics, pattern recognition, and automation of complex tasks.
            """,
            "https://example.com/article2": """
# Latest Technology Trends 2024

The technology landscape in 2024 is dominated by several key trends:

## Generative AI
Large language models and generative AI continue to evolve, offering new 
possibilities for content creation and problem-solving.

## Quantum Computing
Quantum computers are becoming more practical, solving complex problems 
that traditional computers cannot handle efficiently.

## Sustainable Tech
Green technology and sustainable computing practices are becoming 
increasingly important in the tech industry.
            """,
        }

    async def crawl(self, url: str) -> Optional[WebDocumentContent]:
        """Mock 크롤링 결과 반환."""
        content = self.mock_contents.get(
            url,
            f"# Mock Content for {url}\n\nThis is mock content generated for testing purposes.",
        )

        return WebDocumentContent(
            url=url,
            content=content,
            metadata={"mock": True, "crawled_at": datetime.now().isoformat()},
        )


class MockVectorDB(VectorDBAdapter):
    """Mock 벡터 데이터베이스."""

    def __init__(self):
        self.chunks: List[SemanticChunk] = []
        self.embeddings: List[List[float]] = []

    async def add_chunks(self, chunks: List[SemanticChunk]) -> None:
        """Mock 청크 추가."""
        for chunk in chunks:
            # 간단한 Mock 임베딩 생성
            if not chunk.embedding:
                chunk.embedding = [random.random() for _ in range(768)]

            self.chunks.append(chunk)
            self.embeddings.append(chunk.embedding)

    async def search(
        self, query_embedding: List[float], k: int = 10
    ) -> List[Dict[str, Any]]:
        """Mock 유사도 검색."""
        # 간단한 코사인 유사도 계산
        results = []
        for i, chunk in enumerate(self.chunks[:k]):
            # Mock 스코어 생성
            score = random.uniform(0.5, 1.0)
            results.append({"chunk": chunk, "score": score})

        # 스코어로 정렬
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]


class MockPipeline:
    """통합 테스트용 Mock 파이프라인."""

    def __init__(self):
        self.llm_client = MockLLMClient()
        self.search_adapter = MockWebSearchAdapter()
        self.crawling_adapter = MockCrawlingAdapter()
        self.vector_db = MockVectorDB()

    async def run_mock_pipeline(self, query: str) -> Dict[str, Any]:
        """Mock 파이프라인 실행."""
        # 1. 쿼리 생성
        search_queries = await self.llm_client.generate_queries(query)

        # 2. 웹 검색
        all_docs = []
        for sq in search_queries:
            docs = await self.search_adapter.search(sq.processed_queries[0])
            all_docs.extend(docs)

        # 3. 크롤링
        contents = []
        for doc in all_docs[:5]:  # 상위 5개만
            content = await self.crawling_adapter.crawl(doc.url)
            if content:
                contents.append(content)

        # 4. 청킹 (간단한 Mock 구현)
        chunks = []
        for content in contents:
            text = content.content
            chunk_size = 500
            for i in range(0, len(text), chunk_size):
                chunk_text = text[i : i + chunk_size]
                chunk_id = hashlib.md5(f"{content.url}_{i}".encode()).hexdigest()

                chunk = SemanticChunk(
                    chunk_id=chunk_id, content=chunk_text, source_url=content.url
                )
                chunks.append(chunk)

        # 5. 벡터 저장
        await self.vector_db.add_chunks(chunks)

        # 6. 검색
        query_embedding = [random.random() for _ in range(768)]
        search_results = await self.vector_db.search(query_embedding, k=5)

        # 7. 답변 생성
        context = "\n\n".join(
            [
                f"[Source: {r['chunk'].source_url}]\n{r['chunk'].content}"
                for r in search_results
            ]
        )

        answer = await self.llm_client.generate_answer(query, context)

        return {
            "query": query,
            "answer": answer,
            "sources": list(set([r["chunk"].source_url for r in search_results])),
            "num_chunks": len(chunks),
            "num_documents": len(contents),
        }


# tests/test_with_mocks.py
import pytest
from .mocks import (
    MockLLMClient,
    MockWebSearchAdapter,
    MockCrawlingAdapter,
    MockVectorDB,
    MockPipeline,
)


class TestWithMocks:
    """Mock을 사용한 통합 테스트."""

    @pytest.mark.asyncio
    async def test_mock_llm_client(self):
        """Mock LLM 클라이언트 테스트."""
        client = MockLLMClient()

        queries = await client.generate_queries("AI technology")
        assert len(queries) == 2
        assert queries[0].language == "ko"
        assert queries[1].language == "en"

        answer = await client.generate_answer("test", "context")
        assert "test" in answer
        assert len(answer) > 50

    @pytest.mark.asyncio
    async def test_mock_search_adapter(self):
        """Mock 검색 어댑터 테스트."""
        adapter = MockWebSearchAdapter()

        results = await adapter.search("AI", max_results=2)
        assert len(results) == 2
        assert all(r.url.startswith("https://") for r in results)

    @pytest.mark.asyncio
    async def test_mock_crawling_adapter(self):
        """Mock 크롤링 어댑터 테스트."""
        adapter = MockCrawlingAdapter()

        content = await adapter.crawl("https://example.com/article1")
        assert content is not None
        assert "AI and Machine Learning" in content.content
        assert content.url == "https://example.com/article1"

    @pytest.mark.asyncio
    async def test_mock_pipeline_integration(self):
        """Mock 파이프라인 통합 테스트."""
        pipeline = MockPipeline()

        result = await pipeline.run_mock_pipeline("What is AI?")

        assert "query" in result
        assert "answer" in result
        assert "sources" in result
        assert result["num_chunks"] > 0
        assert result["num_documents"] > 0
        assert len(result["sources"]) > 0

    @pytest.mark.asyncio
    async def test_mock_pipeline_performance(self):
        """Mock 파이프라인 성능 테스트."""
        import time

        pipeline = MockPipeline()

        start_time = time.time()
        result = await pipeline.run_mock_pipeline("Latest tech trends")
        end_time = time.time()

        processing_time = end_time - start_time

        # Mock이므로 매우 빠르게 실행되어야 함
        assert processing_time < 1.0
        assert result["answer"] is not None
