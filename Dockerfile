# GraphWeaver Agent - Claude-powered FK Discovery
FROM python:3.14-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 curl build-essential && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy and install
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY mcp_servers/ ./mcp_servers/
COPY agent.py ./
COPY business_rules.yaml ./

# Install dependencies - this will download the embedding model
RUN uv sync

# Pre-download the embedding model to avoid runtime download
RUN uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

ENV PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    HF_HOME=/app/.cache/huggingface

# Run the agent
CMD ["python", "agent.py"]