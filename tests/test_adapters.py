import pytest
from unittest.mock import Mock, AsyncMock, patch
from src.core.models import SemanticChunk
from src.adapters.llm_adapter import VLLMAdapter
from src.adapters.web_search_adapter import TavilySearchAdapter
from src.adapters.vector_store_adapter import FAISSVectorStore


class TestAdapters:
    """어댑터 구현체 테스트."""

    @pytest.mark.asyncio
    async def test_llm_adapter_implements_interface(self):
        """LLM 어댑터가 ILLMService 인터페이스를 구현하는지 테스트."""
        from src.core.models import ILLMService

        adapter = VLLMAdapter()
        assert isinstance(adapter, ILLMService)

        # 인터페이스 메서드 테스트
        queries = await adapter.generate_queries("test query")
        assert len(queries) > 0

        answer = await adapter.generate_answer("test", "context")
        assert isinstance(answer, str)

    @pytest.mark.asyncio
    async def test_vector_store_implements_interface(self):
        """벡터 저장소가 IVectorStore 인터페이스를 구현하는지 테스트."""
        from src.core.models import IVectorStore

        store = FAISSVectorStore(dimension=768)
        assert isinstance(store, IVectorStore)

        # 인터페이스 메서드 테스트
        chunks = [
            SemanticChunk(
                chunk_id=f"chunk_{i}",
                content=f"Content {i}",
                source_url="https://example.com",
                embedding=[0.1 * i] * 768,
            )
            for i in range(5)
        ]

        await store.add_chunks(chunks)
        results = await store.search([0.2] * 768, k=3)
        assert len(results) <= 3

        await store.clear()
        results_after_clear = await store.search([0.2] * 768, k=3)
        assert len(results_after_clear) == 0

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient.post")
    async def test_search_adapter_implements_interface(self, mock_post):
        """검색 어댑터가 IWebSearchService 인터페이스를 구현하는지 테스트."""
        from src.core.models import IWebSearchService

        mock_response = Mock()
        mock_response.json.return_value = {
            "results": [
                {
                    "url": "https://example.com",
                    "title": "Test",
                    "snippet": "Test snippet",
                }
            ]
        }
        mock_post.return_value = mock_response

        adapter = TavilySearchAdapter(api_key="test_key")
        assert isinstance(adapter, IWebSearchService)

        results = await adapter.search("test query", max_results=1)
        assert len(results) == 1
