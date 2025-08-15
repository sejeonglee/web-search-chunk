from typing import List
import httpx
from src.core.models import WebDocument, IWebSearchService


class TavilySearchAdapter(IWebSearchService):
    """Tavily 검색 API 어댑터 - IWebSearchService 구현."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.tavily.com/search"

    async def search(self, query: str, max_results: int = 7) -> List[WebDocument]:
        """Tavily API를 사용한 검색."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.base_url,
                json={"query": query, "max_results": max_results},
                headers={"api-key": self.api_key},
            )
            # 응답 파싱
            results = []
            for item in response.json().get("results", [])[:max_results]:
                results.append(
                    WebDocument(
                        url=item["url"],
                        title=item.get("title"),
                        snippet=item.get("snippet"),
                        search_query=query,
                    )
                )
            return results


class GoogleSearchAdapter(IWebSearchService):
    """Google Search API 어댑터 - IWebSearchService 구현."""

    def __init__(self, api_key: str, cx: str):
        self.api_key = api_key
        self.cx = cx

    async def search(self, query: str, max_results: int = 7) -> List[WebDocument]:
        """Google Search API 구현."""
        # Google Search API 호출 로직
        pass
