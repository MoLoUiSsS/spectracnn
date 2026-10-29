FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install .

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    SPECTRACNN_MODEL_PATH=/app/models/spectracnn.pt \
    SPECTRACNN_META_PATH=/app/models/spectracnn.json \
    SPECTRACNN_WEB_DIR=/app/web

RUN groupadd --system app && useradd --system --gid app --create-home app

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY src ./src
COPY models ./models
COPY web ./web
COPY pyproject.toml README.md LICENSE ./

RUN chown -R app:app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "spectracnn.asgi:app", "--host", "0.0.0.0", "--port", "8000"]
