FROM ghcr.io/astral-sh/uv:0.11.20 AS uv
FROM python:3.12.13-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN groupadd --gid 10001 rag \
    && useradd --uid 10001 --gid rag --create-home --home-dir /home/rag rag

COPY --from=uv /uv /uvx /bin/
WORKDIR /app

# LICENSE is here because pyproject declares license-files: without it the build
# still succeeds but the installed package carries no license text, so the
# published image would claim MIT and ship nothing to back it.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev

COPY alembic.ini ./
COPY migrations ./migrations

RUN chown -R rag:rag /app
USER rag

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
STOPSIGNAL SIGTERM

CMD ["uvicorn", "rag_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
