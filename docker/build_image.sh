#!/bin/bash
# Build the Docker image using latest CUDA base

IMAGE_NAME="chem277b-dev"
TAG="2025.08"

echo "[build] Building Docker image: $IMAGE_NAME:$TAG"
docker build -t $IMAGE_NAME:$TAG .
