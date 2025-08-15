from typing import List, Dict, Any
import numpy as np
import faiss
from src.core.models import SemanticChunk, IVectorStore


class FAISSVectorStore(IVectorStore):
    """FAISS 인메모리 벡터 저장소 - IVectorStore 구현."""

    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.chunks: List[SemanticChunk] = []

    async def add_chunks(self, chunks: List[SemanticChunk]) -> None:
        """청크 벡터 추가."""
        embeddings = []
        for chunk in chunks:
            if chunk.embedding:
                embeddings.append(chunk.embedding)
                self.chunks.append(chunk)

        if embeddings:
            vectors = np.array(embeddings, dtype=np.float32)
            self.index.add(vectors)

    async def search(
        self, query_embedding: List[float], k: int = 10
    ) -> List[Dict[str, Any]]:
        """유사도 기반 검색."""
        query_vector = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.index.search(query_vector, min(k, len(self.chunks)))

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                results.append(
                    {"chunk": self.chunks[idx], "score": float(distances[0][i])}
                )
        return results

    async def clear(self) -> None:
        """저장소 초기화."""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.chunks = []
