from typing import Optional
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import html2text
import asyncio
import random
from src.core.models import WebDocumentContent, ICrawlingService


class PlaywrightCrawler(ICrawlingService):
    """Playwright 기반 크롤러 - ICrawlingService 구현."""

    def __init__(self):
        self.h2t = html2text.HTML2Text()
        self.h2t.ignore_links = True
        self.h2t.ignore_images = True
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        ]

    async def crawl(self, url: str) -> Optional[WebDocumentContent]:
        """웹 페이지 크롤링 및 마크다운 변환."""
        try:
            # 랜덤 지연 (봇 방지)
            await asyncio.sleep(random.uniform(0.5, 2.0))

            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    user_agent=random.choice(self.user_agents)
                )
                page = await context.new_page()

                await page.goto(url, wait_until="networkidle", timeout=10000)
                html_content = await page.content()
                await browser.close()

            # HTML 정제
            soup = BeautifulSoup(html_content, "html.parser")
            for tag in soup(["script", "style", "nav", "footer"]):
                tag.decompose()

            # 마크다운 변환
            markdown = self.h2t.handle(str(soup))

            return WebDocumentContent(
                url=url,
                content=markdown[:50000],  # 길이 제한
            )
        except Exception as e:
            print(f"Crawling error for {url}: {e}")
            return None
