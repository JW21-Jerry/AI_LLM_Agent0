공식 파이썬 3.10 이미지 사용

FROM python:3.10-slim

작업 디렉토리 설정

WORKDIR /app

의존성 파일 복사 및 설치

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

애플리케이션 소스 코드 복사

COPY . .

Cloud Run에서 사용할 포트 노출

EXPOSE 8080

FastAPI 서버 실행 (uvicorn)

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
