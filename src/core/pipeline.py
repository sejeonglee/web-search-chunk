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
        tasks = [
            self.search_service.search(query.processed_queries[0], max_results=7)
            for query in state["search_queries"]
        ]
        results = await asyncio.gather(*tasks)
        state["web_documents"] = [doc for docs in results for doc in docs]
        return state

    async def crawl_documents(self, state: PipelineState) -> PipelineState:
        """문서 크롤링 (병렬 처리)."""
        unique_urls = list({doc.url for doc in state["web_documents"]})
        tasks = [self.crawling_service.crawl(url) for url in unique_urls[:10]]
        contents = await asyncio.gather(*tasks, return_exceptions=True)
        state["document_contents"] = [
            c for c in contents if isinstance(c, WebDocumentContent)
        ]
        return state

    async def chunk_documents(self, state: PipelineState) -> PipelineState:
        """문서 청킹 (배치 처리)."""
        all_chunks = []
        for content in state["document_contents"]:
            chunks = await self.chunking_service.chunk_document(
                content, state["user_query"]
            )
            all_chunks.extend(chunks)
        state["chunks"] = all_chunks
        return state

    async def store_vectors(self, state: PipelineState) -> PipelineState:
        """벡터 저장."""
        await self.vector_store.add_chunks(state["chunks"])
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
