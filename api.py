from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from typing import List
import redis
import os
import uuid
import time

app = FastAPI(title="Memory Hackathon API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

qdrant = None
cache = None
model = None

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class SearchResult(BaseModel):
    text: str
    score: float
    document_id: str

@app.on_event("startup")
async def startup():
    global qdrant, cache, model
    
    print("Starting services...")
    
    qdrant = QdrantClient(host=os.getenv("QDRANT_HOST", "localhost"), port=6333)
    
    collections = qdrant.get_collections()
    collection_names = [c.name for c in collections.collections]
    
    if "documents" not in collection_names:
        qdrant.create_collection(
            collection_name="documents",
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        print("Created Qdrant collection")
    
    cache = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=6379, decode_responses=True)
    print("Connected to Redis")
    
    print("Loading embedding model...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print("Model loaded - Ready!")

@app.get("/health")
def health():
    try:
        qdrant.get_collections()
        qdrant_status = "healthy"
    except Exception as e:
        qdrant_status = f"unhealthy: {str(e)}"
    
    try:
        cache.ping()
        redis_status = "healthy"
    except Exception as e:
        redis_status = f"unhealthy: {str(e)}"
    
    return {
        "status": "healthy",
        "services": {
            "qdrant": qdrant_status,
            "redis": redis_status,
            "embedding": "healthy" if model else "not loaded"
        }
    }

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        content = await file.read()
        text = content.decode('utf-8')
        
        chunk_size = 1000
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        print(f"Processing {len(chunks)} chunks...")
        
        embeddings = model.encode(chunks)
        
        doc_id = str(uuid.uuid4())
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embeddings[i].tolist(),
                payload={
                    "text": chunks[i],
                    "document_id": doc_id,
                    "filename": file.filename,
                    "chunk_index": i
                }
            )
            for i in range(len(chunks))
        ]
        
        qdrant.upsert(collection_name="documents", points=points)
        
        print(f"Uploaded {len(chunks)} chunks")
        
        return {
            "document_id": doc_id,
            "filename": file.filename,
            "chunks": len(chunks),
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", response_model=List[SearchResult])
async def search(request: SearchRequest):
    try:
        cache_key = f"search:{request.query}:{request.top_k}"
        cached = cache.get(cache_key)
        if cached:
            print("Cache hit!")
            import json
            return json.loads(cached)
        
        print(f"Searching: {request.query}")
        
        start = time.time()
        query_embedding = model.encode(request.query)
        embed_time = time.time() - start
        
        start = time.time()
        results = qdrant.query_points(
            collection_name="documents",
            query=query_embedding.tolist(),
            limit=request.top_k
        ).points
        search_time = time.time() - start
        
        search_results = [
            SearchResult(
                text=r.payload["text"][:200] + "...",
                score=r.score,
                document_id=r.payload["document_id"]
            )
            for r in results
        ]
        
        import json
        cache.setex(cache_key, 3600, json.dumps([r.dict() for r in search_results]))
        
        print(f"Found {len(results)} results (embed: {embed_time:.3f}s, search: {search_time:.3f}s)")
        
        return search_results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def metrics():
    try:
        collection = qdrant.get_collection("documents")
        
        info = cache.info()
        hits = info.get('keyspace_hits', 0)
        misses = info.get('keyspace_misses', 0)
        total = hits + misses
        hit_rate = (hits / total * 100) if total > 0 else 0
        
        return {
            "documents": {
                "total_vectors": collection.points_count,
            },
            "cache": {
                "hits": hits,
                "misses": misses,
                "hit_rate": f"{hit_rate:.2f}%"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {
        "message": "Memory Hackathon API",
        "version": "1.0.0",
        "endpoints": ["/health", "/upload", "/search", "/metrics"]
    }
