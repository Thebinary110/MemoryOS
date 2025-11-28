from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from typing import List, Optional, Dict, Any
import logging
from src.models import Chunk, SearchResult
import os

logger = logging.getLogger(__name__)

class QdrantService:
    """Service for interacting with Qdrant vector database"""
    
    def __init__(
        self,
        host: str = None,
        port: int = None,
        collection_name: str = "memories"
    ):
        self.host = host or os.getenv("QDRANT_HOST", "localhost")
        self.port = port or int(os.getenv("QDRANT_PORT", 6333))
        self.collection_name = collection_name
        
        logger.info(f"Connecting to Qdrant at {self.host}:{self.port}")
        self.client = QdrantClient(host=self.host, port=self.port)
        
        self._init_collection()
    
    def _init_collection(self):
        """Initialize collection if it doesn't exist"""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.collection_name not in collection_names:
            logger.info(f"Creating collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=768,  # Snowflake Arctic dimension
                    distance=Distance.COSINE
                )
            )
            logger.info("✅ Collection created")
        else:
            logger.info(f"✅ Collection exists: {self.collection_name}")
    
    def upsert_chunks(self, chunks: List[Chunk], document_id: str):
        """Insert or update chunks in Qdrant"""
        logger.info(f"Upserting {len(chunks)} chunks for document {document_id}")
        
        points = []
        for chunk in chunks:
            if chunk.embedding is None:
                logger.warning(f"Chunk {chunk.id} has no embedding, skipping")
                continue
            
            point = PointStruct(
                id=chunk.id,
                vector=chunk.embedding,
                payload={
                    "text": chunk.text,
                    "document_id": document_id,
                    "metadata": chunk.metadata,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char
                }
            )
            points.append(point)
        
        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"✅ Upserted {len(points)} chunks")
    
    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar chunks"""
        
        # Build filter if provided
        query_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(
                    FieldCondition(
                        key=f"metadata.{key}",
                        match=MatchValue(value=value)
                    )
                )
            if conditions:
                query_filter = Filter(must=conditions)
        
        # Search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=query_filter
        )
        
        # Convert to SearchResult
        search_results = []
        for result in results:
            chunk = Chunk(
                id=str(result.id),
                text=result.payload["text"],
                metadata=result.payload.get("metadata", {}),
                start_char=result.payload.get("start_char", 0),
                end_char=result.payload.get("end_char", 0)
            )
            
            search_result = SearchResult(
                chunk=chunk,
                score=result.score,
                document_id=result.payload["document_id"]
            )
            search_results.append(search_result)
        
        return search_results
    
    def delete_document(self, document_id: str):
        """Delete all chunks for a document"""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            )
        )
        logger.info(f"✅ Deleted document: {document_id}")

def get_qdrant_service() -> QdrantService:
    """Get Qdrant service instance"""
    return QdrantService()
