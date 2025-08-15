import pytest
from unittest.mock import AsyncMock, Mock
from src.core.pipeline import QAPipeline, PipelineState
from src.core.models import (
    SearchQuery,
    WebDocument,
    QAResponse,
    ILLMService,
    IWebSearchService,
    ICrawlingService,
    IVectorStore,
    IChunkingService,
    IRetrievalService,
    IRerankingService,
)


class TestPipeline:
    """파이프라인 테스트 - 모든 의존성은 Mock 인터페이스."""

    @pytest.fixture
    def mock_services(self):
        """Mock 서비스들 생성."""
        return {
            "llm": AsyncMock(spec=ILLMService),
            "search": AsyncMock(spec=IWebSearchService),
            "crawling": AsyncMock(spec=ICrawlingService),
            "vector_store": AsyncMock(spec=IVectorStore),
            "chunking": AsyncMock(spec=IChunkingService),
            "retrieval": AsyncMock(spec=IRetrievalService),
            "reranking": AsyncMock(spec=IRerankingService),
        }

    @pytest.fixture
    def pipeline(self, mock_services):
        """Mock 서비스로 파이프라인 생성."""
        return QAPipeline(
            llm_service=mock_services["llm"],
            search_service=mock_services["search"],
            crawling_service=mock_services["crawling"],
            vector_store=mock_services["vector_store"],
            chunking_service=mock_services["chunking"],
            retrieval_service=mock_services["retrieval"],
            reranking_service=mock_services["reranking"],
        )

    @pytest.mark.asyncio
    async def test_generate_search_queries(self, pipeline, mock_services):
        """검색 쿼리 생성 단계 테스트."""
        state = {"user_query": "test query"}

        mock_services["llm"].generate_queries.return_value = [
            SearchQuery(
                original_query="test query", processed_queries=["test", "query"]
            )
        ]

        result = await pipeline.generate_search_queries(state)

        assert "search_queries" in result
        assert len(result["search_queries"]) > 0
        mock_services["llm"].generate_queries.assert_called_once_with("test query")

    @pytest.mark.asyncio
    async def test_search_web(self, pipeline, mock_services):
        """웹 검색 단계 테스트."""
        state = {
            "search_queries": [
                SearchQuery(original_query="test", processed_queries=["test"])
            ]
        }

        mock_services["search"].search.return_value = [
            WebDocument(url="https://example.com", title="Test", search_query="test")
        ]

        result = await pipeline.search_web(state)

        assert "web_documents" in result
        assert len(result["web_documents"]) > 0
        mock_services["search"].search.assert_called()
