from typing import List
import hashlib
from src.core.models import WebDocumentContent, SemanticChunk, IChunkingService, ILLMService
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class SimpleChunkingAdapter(IChunkingService):
    """간단한 청킹 어댑터 - IChunkingService 구현."""

    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    async def chunk_document(
        self, document: WebDocumentContent, query: str
    ) -> List[SemanticChunk]:
        """문서를 의미적 청크로 분할."""
        chunks = []
        content = document.content

        # 슬라이딩 윈도우 청킹
        for i in range(0, len(content), self.chunk_size - self.overlap):
            chunk_text = content[i : i + self.chunk_size]
            if len(chunk_text.strip()) < 50:
                continue

            chunk_id = hashlib.md5(
                f"{document.url}_{i}_{chunk_text[:50]}".encode()
            ).hexdigest()

            chunks.append(
                SemanticChunk(
                    chunk_id=chunk_id,
                    content=chunk_text,
                    source_url=document.url,
                    metadata={
                        "position": i, 
                        "query": query,
                        "parent_document_id": document.document_id,  # 부모 문서 ID
                        "updated_at": document.crawl_datetime.isoformat()  # 크롤링 시점
                    },
                )
            )

        return chunks


class ContextualChunkingAdapter(IChunkingService):
    """Contextual Retrieval 기반 청킹 어댑터."""

    def __init__(self, llm_service: ILLMService, chunk_size: int = 1000, overlap: int = 200):
        self.llm_service = llm_service
        self.chunk_size = chunk_size
        self.overlap = overlap

    async def chunk_document(
        self, document: WebDocumentContent, query: str
    ) -> List[SemanticChunk]:
        """LLM을 사용한 컨텍스트 추가 청킹."""
        logger.info(f"🔍 Contextual Retrieval 청킹 시작: {document.url}")
        
        # 1단계: 기본 청킹
        raw_chunks = self._create_raw_chunks(document.content, document.url)
        logger.debug(f"  기본 청킹 완료: {len(raw_chunks)}개")
        
        if not raw_chunks:
            return []
            
        # 2단계: 각 청크에 컨텍스트 추가 (LLM 사용)
        contextual_chunks = await self._add_context_to_chunks(
            raw_chunks, document.content, query, document
        )
        
        logger.info(f"✅ Contextual 청킹 완료: {len(contextual_chunks)}개")
        return contextual_chunks
    
    def _create_raw_chunks(self, content: str, url: str) -> List[dict]:
        """기본 슬라이딩 윈도우 청킹."""
        chunks = []
        
        for i in range(0, len(content), self.chunk_size - self.overlap):
            chunk_text = content[i : i + self.chunk_size]
            if len(chunk_text.strip()) < 50:
                continue
                
            chunks.append({
                'text': chunk_text,
                'position': i,
                'url': url
            })
                
        return chunks
    
    async def _add_context_to_chunks(self, raw_chunks: List[dict], full_document: str, query: str, document: WebDocumentContent) -> List[SemanticChunk]:
        """각 청크에 컨텍스트 추가 (VLLM batch inference 사용)."""
        logger.debug(f"🤖 {len(raw_chunks)}개 청크에 VLLM 배치로 컨텍스트 추가 중...")
        
        # 모든 청크에 대한 프롬프트 생성
        prompts = []
        for i, chunk_data in enumerate(raw_chunks):
            prompt = self._create_context_prompt(chunk_data['text'], full_document)
            prompts.append(prompt)
        
        try:
            # 개별 LLM 호출을 병렬로 처리
            import asyncio
            logger.info(f"📦 LLM 개별 추론 시작: {len(prompts)}개 프롬프트")
            tasks = [self.llm_service.generate_answer("", prompt) for prompt in prompts]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            contextual_responses = []
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"개별 LLM 호출 실패: {str(result)}")
                    contextual_responses.append("")
                else:
                    contextual_responses.append(result)
                
            logger.info(f"✅ LLM 개별 추론 완료: {len(contextual_responses)}개 응답")
            
        except Exception as e:
            logger.error(f"❌ LLM 추론 실패: {str(e)}")
            # 폴백: 원본 청크들을 그대로 사용
            contextual_responses = [chunk['text'] for chunk in raw_chunks]
        
        # SemanticChunk 객체 생성
        semantic_chunks = []
        for i, (chunk_data, context_response) in enumerate(zip(raw_chunks, contextual_responses)):
            # 컨텍스트 + 원본 청크 결합
            if context_response and context_response.strip():
                contextual_content = f"{context_response.strip()}\\n\\n{chunk_data['text']}"
            else:
                logger.warning(f"  청크 {i+1} 컨텍스트 생성 실패, 원본 사용")
                contextual_content = chunk_data['text']
            
            chunk_id = hashlib.md5(
                f"{chunk_data['url']}_{chunk_data['position']}_{chunk_data['text'][:50]}".encode()
            ).hexdigest()
            
            semantic_chunks.append(
                SemanticChunk(
                    chunk_id=chunk_id,
                    content=contextual_content,  # 컨텍스트가 추가된 컨텐츠
                    source_url=chunk_data['url'],
                    metadata={
                        'position': chunk_data['position'],
                        'query': query,
                        'original_content': chunk_data['text'],  # 원본 컨텐츠도 저장
                        'contextual_retrieval': True,
                        'parent_document_id': document.document_id,  # 부모 문서 ID
                        'updated_at': document.crawl_datetime.isoformat()  # 크롤링 시점
                    }
                )
            )
        
        return semantic_chunks
    
    def _create_context_prompt(self, chunk_text: str, full_document: str) -> str:
        """청크에 대한 컨텍스트 생성 프롬프트 작성."""
        return f"""
<document>
{full_document}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk_text}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purpose of improving search retrieval. The context should be in Korean and help identify what this chunk is about in relation to the full document.

다음 사항을 고려하여 컨텍스트를 작성해주세요:
1. 이 청크가 전체 문서에서 어떤 주제나 섹션에 속하는지
2. 전체 문맥에서 이 정보의 역할이 무엇인지
3. 검색 시 관련 정보를 찾는 데 도움이 되는 키워드

컨텍스트는 1-2문장 내에서 간결하게 작성해주세요.
"""
    
    async def _generate_context_for_chunk(self, chunk_text: str, full_document: str, query: str, chunk_num: int, total_chunks: int) -> str:
        """단일 청크에 대한 컨텍스트 생성."""
        # Anthropic Contextual Retrieval 프롬프트 기반
        prompt = f"""
<document>
{full_document}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk_text}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purpose of improving search retrieval. The context should be in Korean and help identify what this chunk is about in relation to the full document.

다음 사항을 고려하여 컨텍스트를 작성해주세요:
1. 이 청크가 전체 문서에서 어떤 주제나 섹션에 속하는지
2. 전체 문맥에서 이 정보의 역할이 무엇인지
3. 검색 시 관련 정보를 찾는 데 도움이 되는 키워드

컨텍스트는 1-2문장 내에서 간결하게 작성해주세요.
"""
        
        try:
            # LLM을 사용하여 컨텍스트 생성
            context = await self.llm_service.generate_answer(query="", context=prompt)
            
            # 컨텍스트 + 원본 청크 결합
            contextual_chunk = f"{context.strip()}\n\n{chunk_text}"
            
            logger.debug(f"    청크 {chunk_num}/{total_chunks} 컨텍스트 생성 완료")
            return contextual_chunk
            
        except Exception as e:
            logger.error(f"    청크 {chunk_num}/{total_chunks} 컨텍스트 생성 실패: {str(e)}")
            # 실패 시 원본 청크 반환
            return chunk_text
