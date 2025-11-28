from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from typing import List
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import redis
import os
import uuid
import time

app = FastAPI(title="Memory Hackathon API v2")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

REQUEST_COUNT = Counter('api_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('api_request_duration_seconds', 'Request latency', ['endpoint'])
EMBEDDING_LATENCY = Histogram('embedding_generation_seconds', 'Embedding time')
SEARCH_LATENCY = Histogram('search_duration_seconds', 'Search time')
CACHE_HITS = Counter('cache_hits_total', 'Cache hits')
CACHE_MISSES = Counter('cache_misses_total', 'Cache misses')
ACTIVE_VECTORS = Gauge('active_vectors', 'Vectors in DB')

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
    if "documents" not in [c.name for c in collections.collections]:
        qdrant.create_collection(collection_name="documents", vectors_config=VectorParams(size=384, distance=Distance.COSINE))
    cache = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=6379, decode_responses=True)
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print("Ready!")

@app.get("/health")
def health():
    try:
        qdrant.get_collections()
        qdrant_status = "healthy"
    except: qdrant_status = "unhealthy"
    try:
        cache.ping()
        redis_status = "healthy"
    except: redis_status = "unhealthy"
    return {"status": "healthy", "services": {"qdrant": qdrant_status, "redis": redis_status, "embedding": "healthy" if model else "loading"}}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    start = time.time()
    try:
        content = await file.read()
        text = content.decode('utf-8')
        chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
        embeddings = model.encode(chunks)
        doc_id = str(uuid.uuid4())
        points = [PointStruct(id=str(uuid.uuid4()), vector=embeddings[i].tolist(), payload={"text": chunks[i], "document_id": doc_id, "filename": file.filename, "chunk_index": i}) for i in range(len(chunks))]
        qdrant.upsert(collection_name="documents", points=points)
        REQUEST_COUNT.labels(method='POST', endpoint='/upload', status='success').inc()
        collection = qdrant.get_collection("documents")
        ACTIVE_VECTORS.set(collection.points_count)
        return {"document_id": doc_id, "filename": file.filename, "chunks": len(chunks), "status": "success", "time": f"{time.time()-start:.2f}s"}
    except Exception as e:
        REQUEST_COUNT.labels(method='POST', endpoint='/upload', status='error').inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", response_model=List[SearchResult])
async def search(request: SearchRequest):
    start = time.time()
    try:
        cache_key = f"search:{request.query}:{request.top_k}"
        cached = cache.get(cache_key)
        if cached:
            CACHE_HITS.inc()
            import json
            return json.loads(cached)
        CACHE_MISSES.inc()
        query_embedding = model.encode(request.query)
        results = qdrant.query_points(collection_name="documents", query=query_embedding.tolist(), limit=request.top_k).points
        search_results = [SearchResult(text=r.payload["text"][:200]+"...", score=r.score, document_id=r.payload["document_id"]) for r in results]
        import json
        cache.setex(cache_key, 3600, json.dumps([r.dict() for r in search_results]))
        REQUEST_COUNT.labels(method='POST', endpoint='/search', status='success').inc()
        SEARCH_LATENCY.observe(time.time() - start)
        return search_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics/prometheus")
def prometheus_metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/metrics")
def metrics():
    try:
        collection = qdrant.get_collection("documents")
        info = cache.info()
        hits = info.get('keyspace_hits', 0)
        misses = info.get('keyspace_misses', 0)
        total = hits + misses
        hit_rate = (hits/total*100) if total > 0 else 0
        return {
            "documents": {"total_vectors": collection.points_count},
            "cache": {"hits": hits, "misses": misses, "hit_rate": f"{hit_rate:.2f}%"},
            "aws": {"data_bucket": os.getenv('DATA_BUCKET'), "model_bucket": os.getenv('MODEL_BUCKET')}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "Memory Hackathon API v2.0", "version": "2.0.0", "features": ["Prometheus", "Grafana", "AWS S3"], "endpoints": ["/health", "/upload", "/search", "/metrics"]}
