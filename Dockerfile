# graphweaver-agent/Dockerfile
# GraphWeaver Multi-Agent System
FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 curl build-essential && rm -rf /var/lib/apt/lists/*

# Install uv directly (no ghcr.io)
RUN pip install uv

# LAYER 1: Dependencies
COPY pyproject.toml uv.lock* README.md ./
RUN uv sync && uv pip install streamlit tf-keras langchain

# LAYER 2: Bake ML model INTO image
ENV HF_HOME=/app/.cache/huggingface
RUN uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# LAYER 3: Source code
COPY src/ ./src/
COPY mcp_servers/ ./mcp_servers/

# LAYER 4: Agent files (all agents)
COPY agent.py ./
COPY router_agent.py ./
COPY loan_agent.py ./
COPY streamlit_app.py ./
COPY business_rules.yaml ./
COPY debug_logger.py ./

ENV PYTHONUNBUFFERED=1 PATH="/app/.venv/bin:$PATH"

# Default command runs the router agent
CMD ["python", "router_agent.py"]