from sentence_transformers import SentenceTransformer
import numpy as np
import time

def test_embedding_models():
    """Compare embedding models"""
    
    models = {
        "snowflake": "Snowflake/snowflake-arctic-embed-m-v1.5",
        "jina": "jinaai/jina-embeddings-v2-base-en"
    }
    
    test_text = "How to implement authentication in a web application?"
    
    results = {}
    
    for name, model_path in models.items():
        print(f"\nTesting {name}...")
        
        # Load model
        start = time.time()
        model = SentenceTransformer(model_path)
        load_time = time.time() - start
        
        # Generate embedding
        start = time.time()
        embedding = model.encode(test_text)
        embed_time = time.time() - start
        
        results[name] = {
            "dimensions": len(embedding),
            "load_time": load_time,
            "embed_time": embed_time
        }
        
        print(f"✅ Dimensions: {len(embedding)}")
        print(f"✅ Load time: {load_time:.2f}s")
        print(f"✅ Embedding time: {embed_time:.3f}s")
    
    # Recommendation
    print("\n" + "="*50)
    print("RECOMMENDATION: snowflake-arctic-embed-m-v1.5")
    print("="*50)
    
    return results

if __name__ == "__main__":
    test_embedding_models()
