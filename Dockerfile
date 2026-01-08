# graphweaver-agent/Dockerfile
FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 curl build-essential && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# LAYER 1: Dependencies (cached via uv)
COPY pyproject.toml uv.lock* README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync && uv pip install streamlit

# LAYER 2: Bake ML model INTO image (no cache mount = persists in layer)
ENV HF_HOME=/app/.cache/huggingface
RUN uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# LAYER 3: Source code only (fast rebuild)
COPY src/ ./src/
COPY mcp_servers/ ./mcp_servers/
COPY agent.py streamlit_app.py business_rules.yaml debug_logger.py ./

ENV PYTHONUNBUFFERED=1 PATH="/app/.venv/bin:$PATH"

CMD ["python", "agent.py"]
