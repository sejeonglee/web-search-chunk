"""E2E 테스트 (Mock 버전): 실제 서비스 동작 확인"""

import asyncio
import os
from datetime import datetime
from typing import List
from unittest.mock import AsyncMock, patch
from src.main import WebSearchQASystem
from src.core.models import WebDocument, WebDocumentContent, SemanticChunk


# Mock 데이터
MOCK_SEARCH_RESULTS = [
    WebDocument(
        url="https://example.com/amd-ai-chips",
        title="AMD AI 반도체 최신 동향",
        snippet="AMD가 새로운 Instinct 시리즈 AI 가속기를 발표했습니다.",
        search_query="AMD AI 반도체"
    ),
    WebDocument(
        url="https://example.com/intel-ai-trends", 
        title="Intel의 AI 칩 전략",
        snippet="Intel이 Loihi 뉴로모픽 칩으로 AI 시장에 진출합니다.",
        search_query="Intel AI 칩"
    ),
    WebDocument(
        url="https://example.com/qualcomm-ai",
        title="Qualcomm Snapdragon AI 발전",
        snippet="Qualcomm이 모바일 AI 칩 시장을 선도하고 있습니다.",
        search_query="Qualcomm AI"
    )
]

MOCK_CRAWLED_CONTENT = [
    WebDocumentContent(
        url="https://example.com/amd-ai-chips",
        content="""# AMD AI 반도체 최신 동향

AMD는 2024년 새로운 Instinct MI300 시리즈 AI 가속기를 발표했습니다. 
이 제품은 NVIDIA H100과 경쟁하며, HBM3 메모리를 탑재해 대규모 AI 모델 학습에 최적화되었습니다.

## 주요 특징
- 메모리 용량: 192GB HBM3
- 메모리 대역폭: 5.3TB/s
- FP16 성능: 1.3 PFLOPS

AMD는 OpenAI, Microsoft와 파트너십을 통해 AI 생태계 확장을 추진하고 있습니다.
데이터 센터 시장에서 NVIDIA의 독점 구조에 도전하는 중요한 제품으로 평가받고 있습니다.""",
        crawl_datetime=datetime.now()
    ),
    WebDocumentContent(
        url="https://example.com/intel-ai-trends",
        content="""# Intel의 AI 칩 전략

Intel은 AI 시장에서 다각화된 접근을 취하고 있습니다.

## Loihi 뉴로모픽 칩
- 뇌의 구조를 모방한 스파이킹 신경망 지원
- 저전력 AI 추론에 특화
- 엣지 컴퓨팅 및 IoT 분야 타겟

## Gaudi AI 프로세서
- 훈련 및 추론 모두 지원
- AWS, Google Cloud에서 채택
- Habana Labs 인수로 기술력 강화

Intel은 2025년까지 AI 칩 매출 10억 달러 달성을 목표로 하고 있습니다.""",
        crawl_datetime=datetime.now()
    ),
    WebDocumentContent(
        url="https://example.com/qualcomm-ai",
        content="""# Qualcomm Snapdragon AI 발전

Qualcomm은 모바일 AI 칩 시장의 선두주자입니다.

## Snapdragon 8 Gen 3
- Hexagon NPU로 AI 성능 향상
- 온디바이스 생성형 AI 지원
- 초당 1000억 회 연산 처리

## AI 에코시스템 확장
- Qualcomm AI Hub 런칭
- 개발자를 위한 AI 모델 최적화 도구 제공
- Samsung, Xiaomi 등 주요 파트너와 협력

모바일 AI 시장에서 75% 점유율을 보유하며, 엣지 AI 컴퓨팅으로 영역을 확장하고 있습니다.""",
        crawl_datetime=datetime.now()
    )
]


async def run_e2e_test_mock():
    """Mock을 사용한 E2E 테스트"""
    print("🚀 E2E 테스트 시작 (Mock 버전): NVIDIA가 아닌 다른 AI 반도체 기업들의 산업 동향")
    print("=" * 80)
    
    # Mock 설정
    async def mock_search(query: str, max_results: int = 7) -> List[WebDocument]:
        print(f"🔍 Mock 검색 실행: {query}")
        return MOCK_SEARCH_RESULTS
    
    async def mock_crawl(url: str) -> WebDocumentContent:
        print(f"🕷️  Mock 크롤링 실행: {url}")
        for content in MOCK_CRAWLED_CONTENT:
            if content.url == url:
                return content
        return MOCK_CRAWLED_CONTENT[0]  # fallback
    
    # 설정
    config = {
        "llm_model": "qwen3:4b",
        "embedding_model": "bge-large:335m", 
        "vllm_base_url": "http://localhost:11434/v1",
        "search_provider": "tavily",
        "tavily_api_key": "mock_key",  # Mock API 키
        "vector_dimension": 1024,  # bge-large:335m은 1024차원
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "chunking_strategy": "simple",
        "max_processing_time": 30.0,
        "qdrant_path": "./qdrant_db",
    }
    
    # 시스템 초기화
    print("🔧 시스템 초기화 중...")
    system = WebSearchQASystem(config)
    
    # Mock 패치
    with patch.object(system.container.get_search_service(), 'search', side_effect=mock_search), \
         patch.object(system.container.get_crawling_service(), 'crawl', side_effect=mock_crawl):
        
        # 테스트 쿼리
        test_query = "최근 NVIDIA가 아닌 다른 AI 반도체 기업들의 산업 동향"
        
        print(f"📋 테스트 쿼리: {test_query}")
        print("-" * 80)
        
        try:
            # 쿼리 처리
            start_time = datetime.now()
            print("⏳ 쿼리 처리 시작...")
            
            result = await system.process_query(test_query)
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            print(f"⏱️  총 처리 시간: {processing_time:.2f}초")
            print("-" * 80)
            
            if result["success"]:
                response = result["response"]
                print("✅ 쿼리 처리 성공!")
                print(f"📝 답변:")
                print(response.get("answer", "답변이 없습니다."))
                print("-" * 40)
                
                sources = response.get("sources", [])
                if sources:
                    print(f"📚 참조 소스 ({len(sources)}개):")
                    for i, source in enumerate(sources[:5], 1):
                        print(f"  {i}. {source}")
                
                # Qdrant 저장 확인
                print("-" * 40)
                print("💾 Qdrant 저장 상태 확인...")
                
                # 세션 데이터 저장
                await system._save_session_data()
                
                # 저장된 데이터 로드 테스트
                try:
                    stored_chunks = await system.persistent_store.load_session()
                    print(f"✅ Qdrant에 저장된 SemanticChunk 개수: {len(stored_chunks)}")
                    
                    if stored_chunks:
                        print("📄 저장된 청크 샘플:")
                        for i, chunk in enumerate(stored_chunks[:3], 1):
                            print(f"  {i}. ID: {chunk.chunk_id}")
                            print(f"     URL: {chunk.source_url}")
                            print(f"     내용: {chunk.content[:100]}...")
                            print()
                            
                except Exception as e:
                    print(f"❌ Qdrant 데이터 로드 실패: {str(e)}")
                
            else:
                print("❌ 쿼리 처리 실패!")
                print(f"오류: {result.get('error', '알 수 없는 오류')}")
                
        except Exception as e:
            print(f"❌ E2E 테스트 실패: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("=" * 80)
    print("🏁 E2E 테스트 완료")


if __name__ == "__main__":
    asyncio.run(run_e2e_test_mock())