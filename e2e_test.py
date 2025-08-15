"""E2E 테스트: 실제 서비스 동작 확인"""

import asyncio
import os
from datetime import datetime
from dotenv import load_dotenv
from src.main import WebSearchQASystem

# .env 파일 로드
load_dotenv()


async def run_e2e_test():
    """E2E 테스트 실행"""
    print("🚀 E2E 테스트 시작: NVIDIA가 아닌 다른 AI 반도체 기업들의 산업 동향")
    print("=" * 80)

    # 설정
    config = {
        "llm_model": "qwen3:4b",
        "embedding_model": "bge-large:335m",
        "vllm_base_url": "http://localhost:11434/v1",
        "search_provider": "tavily",
        "tavily_api_key": os.getenv("TAVILY_API_KEY", ""),
        "vector_dimension": 1024,  # bge-large:335m은 1024차원
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "chunking_strategy": "simple",
        "max_processing_time": 300.0,  # E2E 테스트는 더 긴 시간 허용
        "qdrant_path": "./qdrant_db",
    }

    # 시스템 초기화
    print("🔧 시스템 초기화 중...")
    system = WebSearchQASystem(config)

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
    asyncio.run(run_e2e_test())
