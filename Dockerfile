FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.21 /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY README.md ./
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000
CMD ["uvicorn", "ai_news_assistant.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]

