# advanced_rag/Dockerfile 핵심 블록
COPY . /tmp/build_ctx

RUN set -eux; \
    if [ -f /tmp/build_ctx/server/requirements.txt ]; then \
        cp /tmp/build_ctx/server/requirements.txt /app/requirements.txt; \
        cp /tmp/build_ctx/server/main.py /app/main.py; \
    elif [ -f /tmp/build_ctx/advanced_rag/server/requirements.txt ]; then \
        cp /tmp/build_ctx/advanced_rag/server/requirements.txt /app/requirements.txt; \
        cp /tmp/build_ctx/advanced_rag/server/main.py /app/main.py; \
    elif [ -f /tmp/build_ctx/requirements.txt ]; then \
        cp /tmp/build_ctx/requirements.txt /app/requirements.txt; \
        cp /tmp/build_ctx/main.py /app/main.py; \
    fi; \
    rm -rf /tmp/build_ctx
