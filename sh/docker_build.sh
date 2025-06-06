#!/bin/bash

TEAM_NAME="teamx"
IMAGE_TAG="latest"
CACHE_IMAGE="${TEAM_NAME}:${IMAGE_TAG}"

echo "Attempting to pull ${CACHE_IMAGE} for build cache..."
docker pull ${CACHE_IMAGE} >/dev/null 2>&1 || true

echo "Building Docker image ${CACHE_IMAGE}..."
docker build \
    --cache-from ${CACHE_IMAGE} \
    -t ${CACHE_IMAGE} \
    .

if [ $? -ne 0 ]; then
    echo "Error: Docker build failed" >&2
    exit 1
fi

echo "Docker image built successfully: ${CACHE_IMAGE}"