import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from src.main import WebSearchQASystem


class TestIntegration:
    """통합 테스트."""

    @pytest.fixture
    def test_system(self):
        """테스트용 시스템 생성."""
        config = {
            "llm_model": "test-model",
            "search_provider": "tavily",
            "tavily_api_key": "test-key",
            "vector_dimension": 768,
            "chunk_size": 100,
            "chunk_overlap": 20,
            "chunking_strategy": "simple",
            "max_processing_time": 5.0,
            "qdrant_path": ":memory:",
        }
        return WebSearchQASystem(config)

    @pytest.mark.asyncio
    async def test_timeout_handling(self, test_system):
        """타임아웃 처리 테스트."""

        # Mock으로 느린 처리 시뮬레이션
        async def slow_process(query):
            await asyncio.sleep(10)
            return None

        test_system.pipeline.run = slow_process

        result = await test_system.process_query("test query")

        assert result["success"] is False
        assert "timeout" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_error_handling(self, test_system):
        """에러 처리 테스트."""

        # Mock으로 에러 발생
        async def error_process(query):
            raise ValueError("Test error")

        test_system.pipeline.run = error_process

        result = await test_system.process_query("test query")

        assert result["success"] is False
        assert "Test error" in result["error"]

    @pytest.mark.asyncio
    async def test_system_with_different_configs(self):
        """다양한 설정으로 시스템 테스트."""
        configs = [
            {
                "search_provider": "tavily",
                "chunking_strategy": "simple",
            },
            {
                "search_provider": "google",
                "chunking_strategy": "contextual",
            },
        ]

        for config in configs:
            config.update(
                {
                    "llm_model": "test-model",
                    "tavily_api_key": "test-key",
                    "google_api_key": "test-key",
                    "google_cx": "test-cx",
                    "vector_dimension": 768,
                    "chunk_size": 100,
                    "chunk_overlap": 20,
                    "max_processing_time": 5.0,
                    "qdrant_path": ":memory:",
                }
            )

            system = WebSearchQASystem(config)
            assert system is not None
            assert system.pipeline is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
