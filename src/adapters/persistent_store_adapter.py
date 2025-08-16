from typing import List
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from src.core.models import SemanticChunk, IPersistentStore


class QdrantPersistentStore(IPersistentStore):
    """Qdrant 영구 저장소 어댑터 - IPersistentStore 구현."""

    def __init__(self, path: str = "./qdrant_db", session_id: str = "default"):
        # Docker compose로 실행되는 Qdrant에 연결
        try:
            self.client = QdrantClient(host="localhost", port=6333)
        except:
            # 로컬 파일 기반 fallback
            self.client = QdrantClient(path=path)
        self.collection_name = f"session_{session_id}"
        self._init_collection()

    def _init_collection(self):
        """컬렉션 초기화 - 기존 컬렉션이 있으면 재사용, 없으면 생성."""
        try:
            # 기존 컬렉션 존재 여부 확인
            collections = self.client.get_collections()
            collection_exists = any(
                collection.name == self.collection_name 
                for collection in collections.collections
            )
            
            if collection_exists:
                print(f"🔄 기존 컬렉션 '{self.collection_name}' 재사용")
            else:
                # 새 컬렉션 생성 (1024차원)
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
                )
                print(f"✅ 새 컬렉션 '{self.collection_name}' 생성 완료 (1024차원)")
                
        except Exception as e:
            print(f"❌ 컬렉션 초기화 실패: {str(e)}")
            pass

    async def save_session(self, chunks: List[SemanticChunk]) -> None:
        """세션 데이터 영구 저장."""
        try:
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
                print(f"💾 {len(points)}개 청크를 '{self.collection_name}' 컬렉션에 저장했습니다.")
        except Exception as e:
            print(f"❌ 세션 데이터 저장 실패: {str(e)}")

    async def load_session(self, limit: int = 1000) -> List[SemanticChunk]:
        """세션 데이터 로드."""
        try:
            results = self.client.scroll(collection_name=self.collection_name, limit=limit)

            chunks = []
            for point in results[0]:
                chunk_data = point.payload
                chunks.append(SemanticChunk(**chunk_data))
            return chunks
        except Exception as e:
            print(f"⚠️ 세션 데이터 로드 실패 (컬렉션이 없거나 비어있음): {str(e)}")
            return []
