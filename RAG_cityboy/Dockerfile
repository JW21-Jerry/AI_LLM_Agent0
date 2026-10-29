FROM python:3.10-slim

WORKDIR /app

# 파이썬 패키지 빌드에 필요한 리눅스 필수 도구 설치
RUN apt-get update && apt-get install -y build-essential gcc

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
