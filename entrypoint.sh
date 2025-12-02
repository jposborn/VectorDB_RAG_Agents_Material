#!/usr/bin/env sh
set -e

MODEL="${OLLAMA_MODEL:-phi3:mini}"

echo "Starting Ollama server..."
ollama serve &
pid=$!

echo "Waiting for Ollama to be ready..."
for i in $(seq 1 30); do
  if curl -sSf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    echo "Ollama is up."
    break
  fi
  echo "  still starting... ($i)"
  sleep 1
done

echo "Pulling model: ${MODEL}"
ollama pull "${MODEL}" || echo "WARNING: failed to pull model ${MODEL}"

echo "Ollama ready with model ${MODEL}."
wait "${pid}"
