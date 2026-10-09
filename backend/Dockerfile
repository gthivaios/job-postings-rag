FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY . .
EXPOSE 8000
CMD ["sh", "-c", "uv run --no-sync python -m scripts.ingest && uv run --no-sync uvicorn app.main:app --host 0.0.0.0 --port 8000"]