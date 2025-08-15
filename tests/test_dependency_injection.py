import pytest
from src.main import DependencyContainer, WebSearchQASystem


class TestDependencyInjection:
    """의존성 주입 테스트."""

    def test_container_creates_instances(self):
        """컨테이너가 인스턴스를 올바르게 생성하는지 테스트."""
        config = {
            "llm_model": "test-model",
            "search_provider": "tavily",
            "tavily_api_key": "test-key",
            "vector_dimension": 768,
            "chunk_size": 100,
            "chunk_overlap": 20,
            "chunking_strategy": "simple",
            "qdrant_path": ":memory:",
        }

        container = DependencyContainer(config)

        # 서비스 인스턴스 생성 확인
        llm_service = container.get_llm_service()
        assert llm_service is not None

        search_service = container.get_search_service()
        assert search_service is not None

        # 싱글턴 패턴 확인
        llm_service2 = container.get_llm_service()
        assert llm_service is llm_service2

    def test_container_builds_pipeline(self):
        """컨테이너가 파이프라인을 올바르게 조립하는지 테스트."""
        config = {
            "llm_model": "test-model",
            "search_provider": "tavily",
            "tavily_api_key": "test-key",
            "vector_dimension": 768,
            "chunk_size": 100,
            "chunk_overlap": 20,
            "chunking_strategy": "simple",
            "qdrant_path": ":memory:",
        }

        container = DependencyContainer(config)
        pipeline = container.build_pipeline()

        assert pipeline is not None
        assert hasattr(pipeline, "run")

    def test_different_implementations_based_on_config(self):
        """설정에 따라 다른 구현체가 선택되는지 테스트."""
        # Tavily 설정
        config_tavily = {
            "search_provider": "tavily",
            "tavily_api_key": "test-key",
        }
        container_tavily = DependencyContainer(config_tavily)
        search_tavily = container_tavily.get_search_service()

        from src.adapters.web_search_adapter import TavilySearchAdapter

        assert isinstance(search_tavily, TavilySearchAdapter)

        # Google 설정 (구현 필요)
        config_google = {
            "search_provider": "google",
            "google_api_key": "test-key",
            "google_cx": "test-cx",
        }
        container_google = DependencyContainer(config_google)
        # search_google = container_google.get_search_service()
        # assert isinstance(search_google, GoogleSearchAdapter)
