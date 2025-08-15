"""Adapter implementations for external services - implements core interfaces."""

# src/adapters/llm_adapter.py
from typing import List
from src.core.models import SearchQuery, ILLMService


class VLLMAdapter(ILLMService):
    """VLLM 기반 LLM 어댑터 - ILLMService 구현."""

    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-chat-hf"):
        self.model_name = model_name
        # vllm 초기화 로직

    async def generate_queries(self, user_query: str) -> List[SearchQuery]:
        """멀티 쿼리 생성 구현."""
        # 실제 VLLM 호출 로직
        queries = [
            SearchQuery(
                original_query=user_query,
                processed_queries=[
                    user_query,
                    f"{user_query} 최신",
                    f"{user_query} 2024",
                ],
            )
        ]
        return queries

    async def generate_answer(self, query: str, context: str) -> str:
        """답변 생성 구현."""
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        # VLLM 호출 로직
        return f"Based on the context, here's the answer to '{query}'..."
