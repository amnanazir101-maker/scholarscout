FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[app]"

# Bake the embedding model into the image so the first research run starts quickly.
RUN python -c "from scholarscout.store import FastEmbedEmbedder; FastEmbedEmbedder()"

COPY app.py ./
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
