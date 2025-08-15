"""Contextual Retrieval 청킹 단위 테스트"""

import asyncio
import os
from dotenv import load_dotenv
from src.adapters.llm_adapter import VLLMAdapter
from src.adapters.chunking_adapter import ContextualChunkingAdapter
from src.core.models import WebDocumentContent
from src.utils.logger import setup_logger, set_global_log_level

# 전체 로그 레벨을 DEBUG로 설정
set_global_log_level("DEBUG")

# .env 파일 로드
load_dotenv()

async def test_contextual_chunking():
    """Contextual Chunking 단위 테스트"""
    print("🧪 Contextual Chunking 테스트 시작")
    print("=" * 60)
    
    # LLM 서비스 초기화
    llm_service = VLLMAdapter(
        model_name="qwen3:4b",
        embedding_model="bge-large:335m",
        base_url="http://localhost:11434/v1"
    )
    
    # Contextual Chunking Adapter 초기화
    chunking_service = ContextualChunkingAdapter(
        llm_service=llm_service,
        chunk_size=500,  # 테스트를 위해 더 작은 청크 사이즈 사용
        overlap=100
    )
    
    # 테스트용 문서 생성
    test_document = WebDocumentContent(
        url="https://test.com/ai-chip-news",
        content="""
AI 반도체 시장 동향

최근 인공지능 반도체 시장에서는 NVIDIA 외에도 다양한 기업들이 두각을 나타내고 있다.

AMD의 경우, Instinct MI 시리즈를 통해 데이터센터용 AI 가속기 시장에 진출하고 있으며, 2024년 상반기에는 MI300X 칩을 출시하여 ChatGPT와 같은 대형 언어모델 훈련에 사용되고 있다.

Intel은 Gaudi 시리즈를 통해 AI 훈련 및 추론 시장에 도전장을 내밀었다. Gaudi3은 특히 대화형 AI 애플리케이션에 최적화되어 있으며, OpenAI 등의 기업들과 파트너십을 맺고 있다.

Qualcomm은 모바일 AI 칩 시장에서 독보적인 위치를 차지하고 있다. Snapdragon 8 Gen 3은 온디바이스 AI 처리에 특화되어 있으며, 스마트폰에서의 실시간 AI 기능을 가능하게 한다.

Google의 TPU (Tensor Processing Unit)는 자사 클라우드 서비스와 AI 연구에 핵심적인 역할을 하고 있으며, 특히 Transformer 모델 학습에 최적화되어 있다.

이러한 다양한 기업들의 경쟁으로 인해 AI 반도체 시장은 더욱 혁신적이고 경쟁적으로 발전하고 있다.
        """.strip(),
        crawl_datetime="2024-12-15T10:00:00"
    )
    
    print(f"📄 테스트 문서 길이: {len(test_document.content)}자")
    print("-" * 60)
    
    # 테스트 쿼리
    test_query = "AI 반도체 기업들의 최신 동향"
    
    try:
        # Contextual Chunking 실행
        chunks = await chunking_service.chunk_document(test_document, test_query)
        
        print(f"✅ 청킹 완료: {len(chunks)}개 청크 생성")
        print("=" * 60)
        
        # 각 청크의 컨텍스트 확인
        for i, chunk in enumerate(chunks):
            print(f"📝 청크 {i+1}:")
            print(f"   ID: {chunk.chunk_id}")
            print(f"   원본 길이: {len(chunk.metadata.get('original_content', ''))}")
            print(f"   컨텍스트 포함 길이: {len(chunk.content)}")
            print(f"   Contextual Retrieval 적용: {chunk.metadata.get('contextual_retrieval', False)}")
            
            # 컨텍스트가 추가된 경우와 원본 비교
            if chunk.metadata.get('contextual_retrieval'):
                original = chunk.metadata.get('original_content', '')
                if len(chunk.content) > len(original):
                    context_part = chunk.content[:len(chunk.content) - len(original)]
                    print(f"   🔍 추가된 컨텍스트: {context_part[:100]}...")
            
            print(f"   📄 내용 (처음 150자): {chunk.content[:150]}...")
            print("-" * 40)
            
    except Exception as e:
        print(f"❌ 테스트 실패: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("🏁 Contextual Chunking 테스트 완료")
    
    # 리소스 정리
    await llm_service.__aexit__(None, None, None)

if __name__ == "__main__":
    asyncio.run(test_contextual_chunking())