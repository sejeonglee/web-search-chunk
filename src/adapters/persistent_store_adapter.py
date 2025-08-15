from typing import List
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from src.core.models import SemanticChunk, IPersistentStore


class QdrantPersistentStore(IPersistentStore):
    """Qdrant 영구 저장소 어댑터 - IPersistentStore 구현."""

    def __init__(self, path: str = "./qdrant_db"):
        # Docker compose로 실행되는 Qdrant에 연결
        try:
            self.client = QdrantClient(host="localhost", port=6333)
        except:
            # 로컬 파일 기반 fallback
            self.client = QdrantClient(path=path)
        self.collection_name = "semantic_chunks"
        self._init_collection()

    def _init_collection(self):
        """컬렉션 초기화."""
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )
        except:
            pass  # 이미 존재

    async def save_session(self, chunks: List[SemanticChunk]) -> None:
        """세션 데이터 영구 저장."""
        points = []
        for i, chunk in enumerate(chunks):
            if chunk.embedding:
                points.append(
                    PointStruct(
                        id=i, vector=chunk.embedding, payload=chunk.model_dump()
                    )
                )

        if points:
            self.client.upsert(collection_name=self.collection_name, points=points)

    async def load_session(self, limit: int = 1000) -> List[SemanticChunk]:
        """세션 데이터 로드."""
        results = self.client.scroll(collection_name=self.collection_name, limit=limit)

        chunks = []
        for point in results[0]:
            chunk_data = point.payload
            chunks.append(SemanticChunk(**chunk_data))
        return chunks
