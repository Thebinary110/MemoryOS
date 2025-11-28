# -*- coding: utf-8 -*-
"""
Test script for basic system functionality
"""

import requests
import time
import json
import sys

BASE_URL = "http://localhost:8000"

def print_section(title):
    """Print section header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def test_health():
    """Test health endpoint"""
    print_section("Testing Health Check")
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"Status: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
        
        assert response.status_code == 200, "Health check failed"
        print("[PASS] Health check passed")
        return True
    except requests.exceptions.ConnectionError:
        print("[FAIL] Could not connect to API. Is it running?")
        print("Run: docker-compose up -d")
        return False
    except Exception as e:
        print(f"[FAIL] Health check error: {e}")
        return False

def test_upload():
    """Test document upload"""
    print_section("Testing Document Upload")
    
    # Create test file
    test_content = """# Authentication Guide

## Introduction
This guide covers authentication implementation in web applications.

## Basic Authentication
Basic authentication uses username and password credentials.
It is simple but not very secure.

## OAuth 2.0
OAuth 2.0 is a more secure authentication framework.
It uses tokens instead of passwords.

## Best Practices
- Always use HTTPS
- Store passwords securely (bcrypt, argon2)
- Implement rate limiting
- Use multi-factor authentication
"""
    
    # Save test file
    import os
    os.makedirs("data/test", exist_ok=True)
    test_file_path = "data/test/test_auth.md"
    
    with open(test_file_path, "w", encoding="utf-8") as f:
        f.write(test_content)
    
    print(f"Created test file: {test_file_path}")
    
    # Upload file
    try:
        with open(test_file_path, "rb") as f:
            files = {"file": ("test_auth.md", f, "text/markdown")}
            response = requests.post(
                f"{BASE_URL}/documents/upload", 
                files=files,
                timeout=30
            )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"Error: {response.text}")
            return None
        
        result = response.json()
        print(json.dumps(result, indent=2))
        
        assert result["chunks_created"] > 0, "No chunks created"
        
        print(f"[PASS] Upload successful - {result['chunks_created']} chunks created")
        
        return result["document_id"]
        
    except Exception as e:
        print(f"[FAIL] Upload error: {e}")
        return None

def test_search(query="How to implement authentication?"):
    """Test search"""
    print_section(f"Testing Search: '{query}'")
    
    payload = {
        "query": query,
        "top_k": 5
    }
    
    try:
        # First search (cache miss)
        print("Executing first search (cache miss)...")
        start = time.time()
        response = requests.post(
            f"{BASE_URL}/search", 
            json=payload,
            timeout=10
        )
        first_duration = time.time() - start
        
        print(f"Status: {response.status_code}")
        print(f"Duration: {first_duration:.3f}s")
        
        if response.status_code != 200:
            print(f"Error: {response.text}")
            return False
        
        results = response.json()
        print(f"\nFound {len(results)} results:")
        
        for i, result in enumerate(results[:3]):  # Show top 3
            print(f"\n--- Result {i+1} (Score: {result['score']:.4f}) ---")
            text_preview = result['chunk']['text'][:150].replace('\n', ' ')
            print(f"Text: {text_preview}...")
        
        # Second search (cache hit)
        print("\nExecuting second search (cache hit)...")
        start = time.time()
        response = requests.post(
            f"{BASE_URL}/search", 
            json=payload,
            timeout=10
        )
        second_duration = time.time() - start
        
        print(f"\n[PASS] Search successful")
        print(f"First search (cache miss): {first_duration:.3f}s")
        print(f"Second search (cache hit): {second_duration:.3f}s")
        
        if second_duration > 0:
            speedup = first_duration / second_duration
            print(f"Cache speedup: {speedup:.2f}x")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Search error: {e}")
        return False

def test_metrics():
    """Test metrics endpoint"""
    print_section("Testing Metrics")
    
    try:
        response = requests.get(f"{BASE_URL}/metrics/summary", timeout=5)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            print(json.dumps(response.json(), indent=2))
            print("[PASS] Metrics working")
            return True
        else:
            print(f"[FAIL] Metrics returned status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"[FAIL] Metrics error: {e}")
        return False

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("  MEMORY HACKATHON - SYSTEM TESTS")
    print("="*60)
    
    results = {
        "health": False,
        "upload": False,
        "search": False,
        "metrics": False
    }
    
    # Test 1: Health
    results["health"] = test_health()
    if not results["health"]:
        print("\n[ERROR] Cannot proceed without healthy API")
        sys.exit(1)
    
    # Test 2: Upload
    doc_id = test_upload()
    results["upload"] = doc_id is not None
    
    if not results["upload"]:
        print("\n[ERROR] Upload failed, skipping search tests")
    else:
        # Wait for indexing
        print("\nWaiting 3 seconds for indexing...")
        time.sleep(3)
        
        # Test 3: Search
        results["search"] = test_search("How to implement authentication?")
        
        if results["search"]:
            # Additional searches
            test_search("OAuth 2.0 security")
            test_search("password best practices")
    
    # Test 4: Metrics
    results["metrics"] = test_metrics()
    
    # Summary
    print("\n" + "="*60)
    print("  TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {test_name.upper()}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("  ALL TESTS PASSED!")
    else:
        print("  SOME TESTS FAILED")
    print("="*60 + "\n")
    
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()
