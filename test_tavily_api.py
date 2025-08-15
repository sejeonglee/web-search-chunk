"""Tavily Search API 단위 테스트"""

import asyncio
import os
from dotenv import load_dotenv
from src.adapters.web_search_adapter import TavilySearchAdapter

# .env 파일 로드
load_dotenv()


async def test_tavily_api():
    """Tavily API 직접 테스트"""
    print("🔍 Tavily Search API 테스트 시작")
    print("=" * 50)
    
    # API 키 확인
    api_key = os.getenv("TAVILY_API_KEY")
    print(f"📋 TAVILY_API_KEY 로드됨: {'✅ 있음' if api_key else '❌ 없음'}")
    
    if not api_key:
        print("❌ TAVILY_API_KEY가 .env 파일에 없습니다!")
        return
    
    if api_key == "your_tavily_api_key_here":
        print("❌ TAVILY_API_KEY가 기본값입니다. 실제 API 키로 변경해주세요!")
        return
    
    print(f"🔑 API 키 앞 10자리: {api_key[:10]}...")
    print("-" * 50)
    
    # Tavily Adapter 생성
    tavily = TavilySearchAdapter(api_key=api_key)
    
    # 테스트 쿼리
    test_queries = [
        "AI semiconductor companies 2024",
        "AMD vs Intel AI chips",
        "Qualcomm Snapdragon AI"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"🔍 테스트 {i}: '{query}'")
        
        try:
            # 검색 실행
            results = await tavily.search(query, max_results=3)
            
            print(f"✅ 검색 성공: {len(results)}개 결과")
            
            for j, doc in enumerate(results, 1):
                print(f"  {j}. 제목: {doc.title}")
                print(f"     URL: {doc.url}")
                print(f"     요약: {doc.snippet[:100]}...")
                print()
                
        except Exception as e:
            print(f"❌ 검색 실패: {str(e)}")
            # 더 자세한 오류 정보
            import traceback
            traceback.print_exc()
        
        print("-" * 30)
    
    print("🏁 Tavily API 테스트 완료")


if __name__ == "__main__":
    asyncio.run(test_tavily_api())