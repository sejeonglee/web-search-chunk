"""Core domain models and port interfaces for WebSearch QA System."""

from datetime import datetime
from typing import List, Optional, Dict, Any, Protocol
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod


# ============= Domain Models =============


class SearchQuery(BaseModel):
    """검색 질의 모델."""

    original_query: str
    processed_queries: List[str] = Field(default_factory=list)
    language: str = "ko"
    timestamp: datetime = Field(default_factory=datetime.now)


class WebDocument(BaseModel):
    """웹 문서 모델."""

    url: str
    title: Optional[str] = None
    snippet: Optional[str] = None
    search_query: str


class WebDocumentContent(BaseModel):
    """크롤링된 웹 문서 콘텐츠."""

    url: str
    content: str  # markdown format
    crawl_datetime: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SemanticChunk(BaseModel):
    """의미적 청크 단위."""

    chunk_id: str
    content: str
    source_url: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)


class ScratchPad(BaseModel):
    """검색 결과 집합."""

    query: str
    chunks: List[SemanticChunk]
    scores: List[float]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QAResponse(BaseModel):
    """최종 응답 모델."""

    query: str
    answer: str
    sources: List[str]
    confidence: float = 0.0
    processing_time: float = 0.0


# ============= Port Interfaces (Abstractions) =============


class ILLMService(ABC):
    """LLM 서비스 인터페이스."""

    @abstractmethod
    async def generate_queries(self, user_query: str) -> List[SearchQuery]:
        """멀티 쿼리 생성."""
        pass

    @abstractmethod
    async def generate_answer(self, query: str, context: str) -> str:
        """답변 생성."""
        pass


class IWebSearchService(ABC):
    """웹 검색 서비스 인터페이스."""

    @abstractmethod
    async def search(self, query: str, max_results: int = 7) -> List[WebDocument]:
        """웹 검색 수행."""
        pass


class ICrawlingService(ABC):
    """크롤링 서비스 인터페이스."""

    @abstractmethod
    async def crawl(self, url: str) -> Optional[WebDocumentContent]:
        """URL 크롤링."""
        pass


class IVectorStore(ABC):
    """벡터 저장소 인터페이스."""

    @abstractmethod
    async def add_chunks(self, chunks: List[SemanticChunk]) -> None:
        """청크 추가."""
        pass

    @abstractmethod
    async def search(
        self, query_embedding: List[float], k: int = 10
    ) -> List[Dict[str, Any]]:
        """유사도 검색."""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """저장소 초기화."""
        pass


class IPersistentStore(ABC):
    """영구 저장소 인터페이스."""

    @abstractmethod
    async def save_session(self, chunks: List[SemanticChunk]) -> None:
        """세션 데이터 저장."""
        pass

    @abstractmethod
    async def load_session(self, limit: int = 1000) -> List[SemanticChunk]:
        """세션 데이터 로드."""
        pass


class IChunkingService(ABC):
    """청킹 서비스 인터페이스."""

    @abstractmethod
    async def chunk_document(
        self, document: WebDocumentContent, query: str
    ) -> List[SemanticChunk]:
        """문서를 청크로 분할."""
        pass


class IRetrievalService(ABC):
    """검색 서비스 인터페이스."""

    @abstractmethod
    async def retrieve(self, query: str, k: int = 20) -> List[SemanticChunk]:
        """관련 청크 검색."""
        pass


class IRerankingService(ABC):
    """리랭킹 서비스 인터페이스."""

    @abstractmethod
    async def rerank(
        self, query: str, chunks: List[SemanticChunk], k: int = 5
    ) -> Dict[str, Any]:
        """청크 리랭킹."""
        pass
