import asyncio
from typing import TypedDict, List, Annotated, Sequence
from datetime import datetime
from langgraph.graph import StateGraph, END

from .models import (
    SearchQuery,
    WebDocument,
    WebDocumentContent,
    SemanticChunk,
    ScratchPad,
    QAResponse,
    ILLMService,
    IWebSearchService,
    ICrawlingService,
    IVectorStore,
    IChunkingService,
    IRetrievalService,
    IRerankingService,
)
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class PipelineState(TypedDict):
    """파이프라인 상태 정의."""

    user_query: str
    search_queries: List[SearchQuery]
    web_documents: List[WebDocument]
    document_contents: List[WebDocumentContent]
    chunks: List[SemanticChunk]
    scratch_pad: ScratchPad
    response: QAResponse
    start_time: float


class QAPipeline:
    """웹 검색 기반 QA 파이프라인 - 인터페이스에만 의존."""

    def __init__(
        self,
        llm_service: ILLMService,
        search_service: IWebSearchService,
        crawling_service: ICrawlingService,
        vector_store: IVectorStore,
        chunking_service: IChunkingService,
        retrieval_service: IRetrievalService,
        reranking_service: IRerankingService,
    ):
        """모든 의존성은 인터페이스로 주입받음."""
        self.llm_service = llm_service
        self.search_service = search_service
        self.crawling_service = crawling_service
        self.vector_store = vector_store
        self.chunking_service = chunking_service
        self.retrieval_service = retrieval_service
        self.reranking_service = reranking_service
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """LangGraph 워크플로우 구성."""
        workflow = StateGraph(PipelineState)

        # 노드 추가
        workflow.add_node("generate_queries", self.generate_search_queries)
        workflow.add_node("search_web", self.search_web)
        workflow.add_node("crawl_documents", self.crawl_documents)
        workflow.add_node("chunk_documents", self.chunk_documents)
        workflow.add_node("store_vectors", self.store_vectors)
        workflow.add_node("retrieve_chunks", self.retrieve_chunks)
        workflow.add_node("generate_answer", self.generate_answer)

        # 엣지 정의
        workflow.set_entry_point("generate_queries")
        workflow.add_edge("generate_queries", "search_web")
        workflow.add_edge("search_web", "crawl_documents")
        workflow.add_edge("crawl_documents", "chunk_documents")
        workflow.add_edge("chunk_documents", "store_vectors")
        workflow.add_edge("store_vectors", "retrieve_chunks")
        workflow.add_edge("retrieve_chunks", "generate_answer")
        workflow.add_edge("generate_answer", END)

        return workflow.compile()

    async def generate_search_queries(self, state: PipelineState) -> PipelineState:
        """검색 질의 생성 (멀티 쿼리 재작성)."""
        queries = await self.llm_service.generate_queries(state["user_query"])
        state["search_queries"] = queries
        return state

    async def search_web(self, state: PipelineState) -> PipelineState:
        """웹 검색 (병렬 처리)."""
        logger.info(f"🔍 검색 쿼리 개수: {len(state['search_queries'])}")
        tasks = [
            self.search_service.search(query.processed_queries[0], max_results=7)
            for query in state["search_queries"]
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_documents = []
        for i, result in enumerate(results):
            if isinstance(result, list):
                logger.info(f"  쿼리 {i+1}: {len(result)}개 문서 발견")
                all_documents.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"  쿼리 {i+1} 검색 실패: {str(result)}")
        
        logger.info(f"📑 총 {len(all_documents)}개 웹 문서 발견")
        state["web_documents"] = all_documents
        return state

    async def crawl_documents(self, state: PipelineState) -> PipelineState:
        """문서 크롤링 (병렬 처리)."""
        unique_urls = list({doc.url for doc in state["web_documents"]})
        logger.info(f"🕷️  크롤링할 URL 개수: {len(unique_urls)}")
        
        if not unique_urls:
            logger.warning("⚠️  크롤링할 URL이 없습니다.")
            state["document_contents"] = []
            return state
            
        tasks = [self.crawling_service.crawl(url) for url in unique_urls[:10]]
        contents = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful_contents = []
        for i, content in enumerate(contents):
            if isinstance(content, WebDocumentContent):
                logger.debug(f"  ✅ URL {i+1} 크롤링 성공: {unique_urls[i]}")
                successful_contents.append(content)
            else:
                logger.error(f"  ❌ URL {i+1} 크롤링 실패: {unique_urls[i]} - {str(content)}")
        
        state["document_contents"] = successful_contents
        logger.info(f"📄 성공적으로 크롤링된 문서: {len(successful_contents)}개")
        return state

    async def chunk_documents(self, state: PipelineState) -> PipelineState:
        """문서 청킹 (제한된 병렬 처리 - 환경변수로 동시 실행 수 조절)."""
        document_contents = state["document_contents"]
        logger.info(f"📄 크롤링된 문서 개수: {len(document_contents)}")
        
        if not document_contents:
            logger.warning("⚠️  청킹할 문서가 없습니다.")
            state["chunks"] = []
            return state
        
        # 환경변수에서 동시 실행 수 가져오기 (기본값: 2)
        import os
        max_concurrent_chunks = int(os.getenv("MAX_CONCURRENT_CHUNKS", "2"))
        logger.info(f"🔧 최대 동시 청킹 문서 수: {max_concurrent_chunks}개")
        
        # Semaphore로 동시 실행 수 제한
        semaphore = asyncio.Semaphore(max_concurrent_chunks)
        
        async def limited_chunk_document(content):
            async with semaphore:
                logger.debug(f"🔧 문서 청킹 시작: {content.url}")
                return await self.chunking_service.chunk_document(content, state["user_query"])
        
        # 제한된 병렬로 청킹 작업 수행
        tasks = [limited_chunk_document(content) for content in document_contents]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_chunks = []
        for i, result in enumerate(results):
            if isinstance(result, list):
                logger.debug(f"  ✅ 문서 {i+1} 청킹 성공: {len(result)}개 chunk 생성")
                all_chunks.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"  ❌ 문서 {i+1} 청킹 실패: {str(result)}")
        
        state["chunks"] = all_chunks
        logger.info(f"📊 총 {len(all_chunks)}개 chunk 생성 완료")
        return state

    async def store_vectors(self, state: PipelineState) -> PipelineState:
        """벡터 저장 (임베딩 생성 포함)."""
        chunks = state["chunks"]
        logger.info(f"🔢 생성된 chunks 개수: {len(chunks)}")
        
        if not chunks:
            logger.warning("⚠️  저장할 chunks가 없습니다.")
            return state
            
        # 임베딩이 없는 chunks에 임베딩 생성
        chunks_without_embedding = [chunk for chunk in chunks if not chunk.embedding]
        if chunks_without_embedding:
            logger.info(f"🔮 {len(chunks_without_embedding)}개 chunk에 임베딩 생성 중...")
            texts = [chunk.content for chunk in chunks_without_embedding]
            
            try:
                embeddings = await self.llm_service.get_embeddings(texts)
                for chunk, embedding in zip(chunks_without_embedding, embeddings):
                    chunk.embedding = embedding
                logger.info("✅ 임베딩 생성 완료")
            except Exception as e:
                logger.error(f"❌ 임베딩 생성 실패: {str(e)}")
                # 임베딩 없이도 저장할 수 있도록 임시 임베딩 생성
                for chunk in chunks_without_embedding:
                    chunk.embedding = [0.0] * 1024  # 임시 더미 임베딩 (bge-large:335m 차원)
                logger.warning("⚠️  더미 임베딩으로 대체했습니다.")
        
        if chunks:
            logger.debug(f"📝 첫 번째 chunk ID: {chunks[0].chunk_id}")
            logger.debug(f"📄 첫 번째 chunk 내용: {chunks[0].content[:100]}...")
            logger.debug(f"🔮 첫 번째 chunk 임베딩 길이: {len(chunks[0].embedding) if chunks[0].embedding else 0}")
            
        await self.vector_store.add_chunks(chunks)
        logger.info(f"💾 Vector store에 {len(chunks)}개 chunk 저장 완료")
        return state

    async def retrieve_chunks(self, state: PipelineState) -> PipelineState:
        """관련 청크 검색 및 리랭킹."""
        retrieved = await self.retrieval_service.retrieve(state["user_query"], k=20)
        reranked = await self.reranking_service.rerank(
            state["user_query"], retrieved, k=5
        )
        state["scratch_pad"] = ScratchPad(
            query=state["user_query"],
            chunks=reranked["chunks"],
            scores=reranked["scores"],
        )
        return state

    async def generate_answer(self, state: PipelineState) -> PipelineState:
        """최종 답변 생성."""
        context = "\n\n".join(
            [
                f"[Source: {chunk.source_url}]\n{chunk.content}"
                for chunk in state["scratch_pad"].chunks
            ]
        )

        answer = await self.llm_service.generate_answer(
            query=state["user_query"], context=context
        )

        state["response"] = QAResponse(
            query=state["user_query"],
            answer=answer,
            sources=[chunk.source_url for chunk in state["scratch_pad"].chunks],
            processing_time=datetime.now().timestamp() - state["start_time"],
        )
        return state

    async def run(self, query: str) -> QAResponse:
        """파이프라인 실행."""
        initial_state = {
            "user_query": query,
            "start_time": datetime.now().timestamp(),
        }

        final_state = await self.graph.ainvoke(initial_state)
        return final_state["response"]
