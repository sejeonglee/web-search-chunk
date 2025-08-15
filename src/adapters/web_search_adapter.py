from typing import List
import httpx
from src.core.models import WebDocument, IWebSearchService
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class TavilySearchAdapter(IWebSearchService):
    """Tavily 검색 API 어댑터 - IWebSearchService 구현."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.tavily.com/search"

    async def search(self, query: str, max_results: int = 7) -> List[WebDocument]:
        """Tavily API를 사용한 검색."""
        async with httpx.AsyncClient() as client:
            payload = {
                "api_key": self.api_key,
                "query": query,
                "search_depth": "basic",
                "include_answer": False,
                "include_images": False,
                "include_raw_content": False,
                "max_results": max_results
            }
            
            logger.debug(f"🔍 Tavily 검색 요청: {query}")
            
            response = await client.post(
                self.base_url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            
            logger.debug(f"📡 Tavily 응답 상태: {response.status_code}")
            response_data = response.json()
            logger.debug(f"📄 Tavily 응답 데이터: {response_data}")
            
            response.raise_for_status()
            
            # 응답 파싱
            results = []
            tavily_results = response_data.get("results", [])
            logger.info(f"📊 Tavily 검색 결과: {len(tavily_results)}개 문서 발견")
            
            for i, item in enumerate(tavily_results[:max_results]):
                logger.debug(f"  결과 {i+1}: {item.get('title', 'No Title')}")
                results.append(
                    WebDocument(
                        url=item["url"],
                        title=item.get("title", "No Title"),
                        snippet=item.get("content", "No Content"),
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
