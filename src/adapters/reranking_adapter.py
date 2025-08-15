from typing import List, Dict, Any
from src.core.models import SemanticChunk, IRerankingService


class CrossEncoderRerankingAdapter(IRerankingService):
    """Cross-encoder 기반 리랭킹 어댑터 - IRerankingService 구현."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        # Cross-encoder 모델 초기화

    async def rerank(
        self, query: str, chunks: List[SemanticChunk], k: int = 5
    ) -> Dict[str, Any]:
        """청크 리랭킹."""
        # Cross-encoder 스코어링
        scored_chunks = []
        for chunk in chunks:
            score = self._calculate_relevance(query, chunk.content)
            scored_chunks.append((chunk, score))

        # 스코어로 정렬
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        return {
            "chunks": [c[0] for c in scored_chunks[:k]],
            "scores": [c[1] for c in scored_chunks[:k]],
        }

    def _calculate_relevance(self, query: str, content: str) -> float:
        """관련성 점수 계산."""
        # 실제 cross-encoder 구현 필요
        query_terms = set(query.lower().split())
        content_terms = set(content.lower().split())

        if not query_terms:
            return 0.0

        overlap = len(query_terms & content_terms)
        return overlap / len(query_terms)
