from typing import List
import hashlib
from src.core.models import WebDocumentContent, SemanticChunk, IChunkingService


class SimpleChunkingAdapter(IChunkingService):
    """간단한 청킹 어댑터 - IChunkingService 구현."""

    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    async def chunk_document(
        self, document: WebDocumentContent, query: str
    ) -> List[SemanticChunk]:
        """문서를 의미적 청크로 분할."""
        chunks = []
        content = document.content

        # 슬라이딩 윈도우 청킹
        for i in range(0, len(content), self.chunk_size - self.overlap):
            chunk_text = content[i : i + self.chunk_size]
            if len(chunk_text.strip()) < 50:
                continue

            chunk_id = hashlib.md5(
                f"{document.url}_{i}_{chunk_text[:50]}".encode()
            ).hexdigest()

            chunks.append(
                SemanticChunk(
                    chunk_id=chunk_id,
                    content=chunk_text,
                    source_url=document.url,
                    metadata={"position": i, "query": query},
                )
            )

        return chunks


class ContextualChunkingAdapter(IChunkingService):
    """Contextual Retrieval 기반 청킹 어댑터."""

    def __init__(self, llm_service, chunk_size: int = 1000):
        self.llm_service = llm_service
        self.chunk_size = chunk_size

    async def chunk_document(
        self, document: WebDocumentContent, query: str
    ) -> List[SemanticChunk]:
        """LLM을 사용한 컨텍스트 추가 청킹."""
        # Anthropic Contextual Retrieval 구현
        pass
