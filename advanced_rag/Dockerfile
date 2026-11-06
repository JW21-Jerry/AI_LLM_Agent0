# ==============================================================================
# Google Cloud Run 배포용 Dockerfile (advanced_rag 루트 및 server 폴더 모두 호환)
# ==============================================================================
FROM python:3.11-slim

# 시스템 패키지 설치 (FAISS 구동에 필요한 libgomp1 등)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 빌드 컨텍스트가 프로젝트 루트(/workspace), advanced_rag(/workspace/advanced_rag),
# 또는 server(/workspace/advanced_rag/server) 중 어디로 설정되더라도 자동 탐색하여 복사
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
    else \
        echo "ERROR: requirements.txt or main.py not found in build context" >&2; \
        exit 1; \
    fi; \
    rm -rf /tmp/build_ctx

# 의존성 설치
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Cloud Run 기본 포트 환경 변수 설정
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

# FastAPI 서버 실행
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]

