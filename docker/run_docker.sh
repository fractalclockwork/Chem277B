#!/bin/bash

# Container configuration
IMAGE_NAME="chem277b-dev"
TAG="2025.08"
CONTAINER_NAME="chem277b-session"
PORT=8888

# Host user info
USERNAME=$(id -un)
USER_ID=$(id -u)
GROUP_ID=$(id -g)
CONTAINER_HOME="/home/$USERNAME"

# Require workspace path as argument
if [ -z "$1" ]; then
  echo "[error] No workspace path provided."
  echo "Usage: source run_docker.sh /absolute/path/to/workspace"
  exit 1
fi

HOST_WORKSPACE="$1"

# Validate workspace path
if [ ! -d "$HOST_WORKSPACE" ]; then
  echo "[error] Workspace path does not exist: $HOST_WORKSPACE"
  exit 1
fi

# Mirror path inside container
CONTAINER_WORKSPACE="$CONTAINER_HOME$(echo "$HOST_WORKSPACE" | sed "s|$HOME||")"

echo "[run] Launching container: $CONTAINER_NAME"
echo "[info] Mounting workspace: $HOST_WORKSPACE → $CONTAINER_WORKSPACE"

docker run -it --rm \
  --gpus all \
  --name $CONTAINER_NAME \
  -p $PORT:$PORT \
  -v "$HOST_WORKSPACE":"$CONTAINER_WORKSPACE" \
  -v "$HOME":"$CONTAINER_HOME" \
  -v "$HOME/.ssh":"/root/.ssh" \
  -v "$HOME/.gitconfig":"/root/.gitconfig" \
  --user $USER_ID:$GROUP_ID \
  --env HOME="$CONTAINER_HOME" \
  --workdir "$CONTAINER_WORKSPACE" \
  --entrypoint bash \
  $IMAGE_NAME:$TAG \
  -c "mkdir -p \$HOME/.local/share/jupyter/runtime && exec jupyter lab --ip=0.0.0.0 --port=$PORT --no-browser"

