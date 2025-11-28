from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
import logging
from datetime import datetime
import uuid
import os

from src.models import (
    Document, DocumentType, SearchRequest, SearchResult, 
    HealthResponse, Chunk
)
from src.embeddings.service import get_embedding_service
from src.chunking.service import get_chunking_service
from src.storage.qdrant_service import get_qdrant_service
from src.api.cache import get_cache_service

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Memory Hackathon API",
    description="AI Memory System with Advanced Retrieval",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global services (initialized on startup)
embedding_service = None
chunking_service = None
qdrant_service = None
cache_service = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global embedding_service, chunking_service, qdrant_service, cache_service
    
    logger.info("Ì∫Ä Starting Memory Hackathon API...")
    
    # Initialize services
    embedding_service = get_embedding_service()
    chunking_service = get_chunking_service()
    qdrant_service = get_qdrant_service()
    cache_service = get_cache_service()
    
    logger.info("‚úÖ All services initialized!")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Check Qdrant
        qdrant_health = "healthy"
        try:
            qdrant_service.client.get_collections()
        except Exception as e:
            qdrant_health = f"unhealthy: {str(e)}"
        
        # Check Redis
        redis_health = "healthy"
        try:
            cache_service.redis.ping()
        except Exception as e:
            redis_health = f"unhealthy: {str(e)}"
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now(),
            services={
                "qdrant": qdrant_health,
                "redis": redis_health,
                "embedding": "healthy"
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Upload and process a document
    
    Steps:
    1. Save file
    2. Extract text
    3. Chunk text
    4. Generate embeddings
    5. Store in Qdrant
    """
    try:
        # Generate document ID
        doc_id = str(uuid.uuid4())
        
        # Determine file type
        file_ext = os.path.splitext(file.filename)[1].lower()
        file_type_map = {
            '.pdf': DocumentType.PDF,
            '.txt': DocumentType.TEXT,
            '.md': DocumentType.MARKDOWN,
            '.py': DocumentType.CODE,
            '.js': DocumentType.CODE,
        }
        
        file_type = file_type_map.get(file_ext, DocumentType.TEXT)
        
        # Save file temporarily
        upload_dir = "/app/data/uploads"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = f"{upload_dir}/{doc_id}_{file.filename}"
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"Ì≥Ñ File saved: {file_path}")
        
        # Extract text (simplified - just read for now)
        if file_type == DocumentType.PDF:
            # TODO: Implement PDF extraction
            # For now, return error
            raise HTTPException(
                status_code=400, 
                detail="PDF processing not yet implemented. Use .txt or .md files for now."
            )
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
        
        logger.info(f"‚úÖ Extracted {len(text_content)} characters")
        
        # Chunk text
        chunks = chunking_service.chunk_text(
            text_content,
            metadata={
                "filename": file.filename,
                "file_type": file_type.value,
                "upload_date": datetime.now().isoformat()
            }
        )
        
        logger.info(f"‚úÖ Created {len(chunks)} chunks")
        
        # Generate embeddings (batch)
        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = embedding_service.embed_batch(chunk_texts)
        
        # Assign embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        
        logger.info(f"‚úÖ Generated {len(embeddings)} embeddings")
        
        # Store in Qdrant
        qdrant_service.upsert_chunks(chunks, doc_id)
        
        # Clean up file
        os.remove(file_path)
        
        return {
            "document_id": doc_id,
            "filename": file.filename,
            "file_type": file_type.value,
            "chunks_created": len(chunks),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", response_model=List[SearchResult])
async def search(request: SearchRequest):
    """
    Search for similar chunks
    
    With caching:
    1. Check cache for query
    2. If cached, return
    3. If not, generate embedding
    4. Search Qdrant
    5. Cache result
    6. Return
    """
    try:
        # Check cache
        cache_key = f"search:{request.query}:{request.top_k}"
        cached_result = cache_service.get(cache_key)
        
        if cached_result:
            logger.info(f"‚úÖ Cache hit for query: {request.query}")
            return cached_result
        
        logger.info(f"Ì¥ç Searching for: {request.query}")
        
        # Generate query embedding
        query_embedding = embedding_service.embed_text(request.query)
        
        # Search Qdrant
        results = qdrant_service.search(
            query_vector=query_embedding,
            top_k=request.top_k,
            filters=request.filters
        )
        
        # Cache result
        cache_service.set(cache_key, results, ttl=3600)  # Cache for 1 hour
        
        logger.info(f"‚úÖ Found {len(results)} results")
        
        return results
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and all its chunks"""
    try:
        qdrant_service.delete_document(document_id)
        return {"status": "success", "document_id": document_id}
    except Exception as e:
        logger.error(f"Delete error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get system metrics"""
    try:
        # Get collection info
        collection_info = qdrant_service.client.get_collection(
            collection_name=qdrant_service.collection_name
        )
        
        # Get cache stats
        cache_stats = cache_service.get_stats()
        
        return {
            "qdrant": {
                "vectors_count": collection_info.vectors_count,
                "points_count": collection_info.points_count,
            },
            "cache": cache_stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

from src.api.metrics import MetricsCollector, track_request_metrics, CACHE_HITS, CACHE_MISSES

@app.get("/metrics/prometheus")
async def prometheus_metrics():
    """Prometheus metrics endpoint"""
    from prometheus_client import generate_latest
    return generate_latest()

@app.get("/metrics/summary")
async def metrics_summary():
    """Human-readable metrics summary"""
    return MetricsCollector.get_metrics_summary()
