import pytest
from datetime import datetime
from src.core.models import (
    SearchQuery,
    WebDocument,
    WebDocumentContent,
    SemanticChunk,
    ScratchPad,
    QAResponse,
)


class TestModels:
    """도메인 모델 테스트."""

    def test_search_query_creation(self):
        """SearchQuery 모델 생성 테스트."""
        query = SearchQuery(
            original_query="test query",
            processed_queries=["test", "query"],
            language="ko",
        )
        assert query.original_query == "test query"
        assert len(query.processed_queries) == 2
        assert query.language == "ko"
        assert isinstance(query.timestamp, datetime)

    def test_web_document_creation(self):
        """WebDocument 모델 생성 테스트."""
        doc = WebDocument(
            url="https://example.com",
            title="Test Title",
            snippet="Test snippet",
            search_query="test",
        )
        assert doc.url == "https://example.com"
        assert doc.title == "Test Title"

    def test_semantic_chunk_creation(self):
        """SemanticChunk 모델 생성 테스트."""
        chunk = SemanticChunk(
            chunk_id="test_id", content="Test content", source_url="https://example.com"
        )
        assert chunk.chunk_id == "test_id"
        assert chunk.content == "Test content"
        assert chunk.embedding is None
