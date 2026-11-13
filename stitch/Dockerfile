FROM python:3.11-slim

WORKDIR /app

COPY . /tmp/build_ctx

RUN set -eux; \
    if [ -f /tmp/build_ctx/requirements.txt ]; then \
        cp /tmp/build_ctx/requirements.txt /app/requirements.txt; \
        cp /tmp/build_ctx/main.py /app/main.py; \
    elif [ -f /tmp/build_ctx/server/requirements.txt ]; then \
        cp /tmp/build_ctx/server/requirements.txt /app/requirements.txt; \
        cp /tmp/build_ctx/server/main.py /app/main.py; \
    elif [ -f /tmp/build_ctx/AI_deep_class/fashionRAG/server/requirements.txt ]; then \
        cp /tmp/build_ctx/AI_deep_class/fashionRAG/server/requirements.txt /app/requirements.txt; \
        cp /tmp/build_ctx/AI_deep_class/fashionRAG/server/main.py /app/main.py; \
    fi; \
    rm -rf /tmp/build_ctx

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

ENV PORT=8080
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]

