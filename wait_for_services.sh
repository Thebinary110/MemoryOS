#!/bin/bash

echo "Waiting for services to be ready..."

# Wait for Qdrant
echo -n "Waiting for Qdrant..."
for i in {1..30}; do
    if curl -s http://localhost:6333/health > /dev/null 2>&1; then
        echo " Ready!"
        break
    fi
    echo -n "."
    sleep 2
done

# Wait for Redis
echo -n "Waiting for Redis..."
for i in {1..30}; do
    if docker exec redis redis-cli ping > /dev/null 2>&1; then
        echo " Ready!"
        break
    fi
    echo -n "."
    sleep 2
done

# Wait for App
echo -n "Waiting for App..."
for i in {1..60}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo " Ready!"
        break
    fi
    echo -n "."
    sleep 2
done

echo "All services ready!"
