"""Adapters for external services."""

from .llm_client import LLMClient, VLLMClient
from .web_search_adapter import WebSearchAdapter, TavilySearchAdapter
from .crawling_adapter import CrawlingAdapter, PlaywrightCrawler
from .vector_db_adapter import VectorDBAdapter, FAISSAdapter
from .permanent_db_adapter import PermanentDBAdapter

__all__ = [
    "LLMClient",
    "VLLMClient",
    "WebSearchAdapter",
    "TavilySearchAdapter",
    "CrawlingAdapter",
    "PlaywrightCrawler",
    "VectorDBAdapter",
    "FAISSAdapter",
    "PermanentDBAdapter",
]
