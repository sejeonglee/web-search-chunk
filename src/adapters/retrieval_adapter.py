from typing import List, Dict, Any
import re
import math
from collections import Counter
from src.core.models import SemanticChunk, IRetrievalService, IVectorStore


class HybridRetrievalAdapter(IRetrievalService):
    """하이브리드 검색 어댑터 - IRetrievalService 구현."""

    def __init__(self, vector_store: IVectorStore, llm_service=None):
        self.vector_store = vector_store
        self.llm_service = llm_service  # LLM 서비스 (임베딩 생성용)

    async def retrieve(self, query: str, k: int = 20) -> List[SemanticChunk]:
        """하이브리드 검색 수행."""
        # 1. 임베딩 생성
        if self.llm_service:
            try:
                embeddings = await self.llm_service.get_embeddings([query])
                query_embedding = embeddings[0]
                print(f"🔮 쿼리 임베딩 생성 완료: {len(query_embedding)}차원")
            except Exception as e:
                print(f"❌ 쿼리 임베딩 생성 실패: {str(e)}")
                query_embedding = [0.1] * 1024  # 더미 임베딩 (1024차원)
        else:
            query_embedding = [0.1] * 1024  # 더미 임베딩 (1024차원)

        # 2. 벡터 검색
        vector_results = await self.vector_store.search(query_embedding, k)

        # 3. BM25 검색 (간단한 구현)
        keyword_results = await self._bm25_search(query, k)

        # 4. RRF 융합
        combined = self._reciprocal_rank_fusion(vector_results, keyword_results)

        return [item["chunk"] for item in combined[:k]]

    async def _bm25_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """BM25 기반 검색."""
        # Vector store에서 모든 chunks 가져오기
        if not hasattr(self.vector_store, 'chunks'):
            return []
        
        chunks = getattr(self.vector_store, 'chunks', [])
        
        # 쿼리 토큰화
        query_tokens = self._tokenize(query.lower())
        if not query_tokens:
            return []
        
        # 문서별 토큰화 및 통계 계산
        documents = []
        doc_lengths = []
        
        for chunk in chunks:
            tokens = self._tokenize(chunk.content.lower())
            documents.append(tokens)
            doc_lengths.append(len(tokens))
        
        if not documents:
            return []
        
        # 평균 문서 길이
        avgdl = sum(doc_lengths) / len(doc_lengths)
        
        # 각 쿼리 토큰이 등장하는 문서 수 계산 (IDF 계산용)
        df = {}  # document frequency
        for tokens in documents:
            unique_tokens = set(tokens)
            for token in query_tokens:
                if token in unique_tokens:
                    df[token] = df.get(token, 0) + 1
        
        # BM25 파라미터
        k1 = 1.2
        b = 0.75
        N = len(documents)
        
        scores = []
        
        for i, tokens in enumerate(documents):
            score = 0.0
            token_counts = Counter(tokens)
            doc_len = doc_lengths[i]
            
            for token in query_tokens:
                if token in token_counts:
                    tf = token_counts[token]  # term frequency
                    idf = math.log((N - df.get(token, 0) + 0.5) / (df.get(token, 0) + 0.5))
                    
                    # BM25 공식
                    numerator = idf * tf * (k1 + 1)
                    denominator = tf + k1 * (1 - b + b * (doc_len / avgdl))
                    score += numerator / denominator
            
            scores.append((i, score))
        
        # 점수순으로 정렬
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # 상위 k개 결과 반환
        results = []
        for i, score in scores[:k]:
            if score > 0:  # 점수가 0보다 큰 경우만
                results.append({
                    "chunk": chunks[i],
                    "score": score
                })
        
        return results
    
    def _tokenize(self, text: str) -> List[str]:
        """텍스트 토큰화 (간단한 구현)."""
        # 알파벳, 숫자, 한글만 추출하고 소문자로 변환
        tokens = re.findall(r'[a-zA-Z0-9가-힣]+', text)
        # 2글자 이상인 토큰만 반환
        return [token for token in tokens if len(token) >= 2]

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
