"""Adapter implementations for external services - implements core interfaces."""

# src/adapters/llm_adapter.py
from typing import List
import httpx
import json
from src.core.models import SearchQuery, ILLMService


class VLLMAdapter(ILLMService):
    """VLLM 서빙 LLM 어댑터 - ILLMService 구현."""

    def __init__(
        self, 
        model_name: str = "qwen3:4b",
        embedding_model: str = "bge-large:335m",
        base_url: str = "http://localhost:8000/v1"
    ):
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)

    async def generate_queries(self, user_query: str) -> List[SearchQuery]:
        """멀티 쿼리 생성 구현."""
        prompt = f"""다음 사용자 질문을 분석하여 웹 검색에 적합한 3개의 다양한 검색 쿼리를 생성해주세요.
원본 질문: {user_query}

각 쿼리는 서로 다른 관점이나 키워드를 사용해야 합니다.

응답 형식:
1. [첫 번째 검색 쿼리]
2. [두 번째 검색 쿼리]  
3. [세 번째 검색 쿼리]"""

        try:
            response = await self._call_vllm(prompt)
            # 응답에서 쿼리 추출
            lines = response.strip().split('\n')
            processed_queries = []
            for line in lines:
                if line.strip() and any(line.startswith(f"{i}.") for i in range(1, 10)):
                    query = line.split('.', 1)[1].strip()
                    if query:
                        processed_queries.append(query)
            
            # 최소 1개 쿼리는 보장
            if not processed_queries:
                processed_queries = [user_query]
                
            return [SearchQuery(
                original_query=user_query,
                processed_queries=processed_queries[:3]  # 최대 3개
            )]
            
        except Exception as e:
            # 에러 시 기본 쿼리 반환
            return [SearchQuery(
                original_query=user_query,
                processed_queries=[user_query]
            )]

    async def generate_answer(self, query: str, context: str) -> str:
        """답변 생성 구현."""
        prompt = f"""다음 컨텍스트를 바탕으로 사용자 질문에 대해 정확하고 유용한 답변을 제공해주세요.

컨텍스트:
{context}

질문: {query}

답변은 다음 조건을 만족해야 합니다:
1. 컨텍스트에 기반한 정확한 정보 제공
2. 명확하고 이해하기 쉬운 한국어
3. 가능한 한 구체적인 정보 포함
4. 출처가 불분명한 정보는 포함하지 않음

답변:"""

        try:
            return await self._call_vllm(prompt)
        except Exception as e:
            return f"죄송합니다. 답변 생성 중 오류가 발생했습니다: {str(e)}"

    async def _call_vllm(self, prompt: str) -> str:
        """VLLM 서빙 API 호출."""
        try:
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "stream": False,
                "temperature": 0.1,
                "max_tokens": 1024
            }
            
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except Exception as e:
            raise Exception(f"VLLM API 호출 실패: {str(e)}")

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """텍스트 임베딩 생성 (VLLM 임베딩 서빙)."""
        try:
            embeddings = []
            for text in texts:
                payload = {
                    "model": self.embedding_model,
                    "input": text
                }
                
                response = await self.client.post(
                    f"{self.base_url}/embeddings",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                
                result = response.json()
                embeddings.append(result["data"][0]["embedding"])
                
            return embeddings
            
        except Exception as e:
            raise Exception(f"임베딩 생성 실패: {str(e)}")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
