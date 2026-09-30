# graphwalk MCP server over streamable HTTP.
#
#   docker run -p 8080:8080 -v graphwalk-data:/data \
#     -e GRAPHWALK_AUTH=bearer -e GRAPHWALK_AUTH_TOKEN=... \
#     -e GRAPHWALK_ALLOWED_HOSTS=graphwalk.example.com \
#     ghcr.io/<owner>/graphwalk
#
# Clients send provider keys as headers (X-OpenRouter-API-Key, ...). The server refuses
# to start without GRAPHWALK_AUTH=bearer or oauth, since it binds to 0.0.0.0.

FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.8.17 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --locked --no-dev --no-install-project --extra mcp --extra llm --extra embeddings
COPY src ./src
RUN uv sync --locked --no-dev --no-editable --extra mcp --extra llm --extra embeddings

FROM python:3.12-slim
RUN useradd --create-home --uid 10001 graphwalk \
    && mkdir /data && chown graphwalk:graphwalk /data
COPY --from=build /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    LITELLM_LOCAL_MODEL_COST_MAP=True \
    GRAPHWALK_DB=/data/graphwalk.db \
    GRAPHWALK_HOST=0.0.0.0 \
    GRAPHWALK_PORT=8080
USER graphwalk
WORKDIR /data
VOLUME /data
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=4)"]
ENTRYPOINT ["graphwalk", "mcp", "--http"]
