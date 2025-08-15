from typing import List, Dict, Any
from src.core.models import SemanticChunk, IRetrievalService, IVectorStore


class HybridRetrievalAdapter(IRetrievalService):
    """하이브리드 검색 어댑터 - IRetrievalService 구현."""

    def __init__(self, vector_store: IVectorStore, embedder=None):
        self.vector_store = vector_store
        self.embedder = embedder  # 임베딩 모델

    async def retrieve(self, query: str, k: int = 20) -> List[SemanticChunk]:
        """하이브리드 검색 수행."""
        # 1. 임베딩 생성 (실제로는 임베딩 모델 사용)
        query_embedding = [0.1] * 768  # 더미 구현

        # 2. 벡터 검색
        vector_results = await self.vector_store.search(query_embedding, k)

        # 3. BM25 검색 (간단한 구현)
        keyword_results = await self._bm25_search(query, k)

        # 4. RRF 융합
        combined = self._reciprocal_rank_fusion(vector_results, keyword_results)

        return [item["chunk"] for item in combined[:k]]

    async def _bm25_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """BM25 기반 검색."""
        # 실제 BM25 구현 필요
        return []

    def _reciprocal_rank_fusion(
        self, vector_results: List[Dict], keyword_results: List[Dict], k: int = 60
    ) -> List[Dict]:
        """RRF를 사용한 결과 융합."""
        scores = {}

        for rank, item in enumerate(vector_results):
            chunk_id = item["chunk"].chunk_id
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)

        for rank, item in enumerate(keyword_results):
            chunk_id = item["chunk"].chunk_id
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)

        # 결과 정렬 및 반환
        sorted_chunks = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        chunk_map = {
            item["chunk"].chunk_id: item for item in vector_results + keyword_results
        }

        result = []
        for chunk_id, score in sorted_chunks:
            if chunk_id in chunk_map:
                result.append(chunk_map[chunk_id])

        return result
