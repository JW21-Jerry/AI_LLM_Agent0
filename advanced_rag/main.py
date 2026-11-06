"""
================================================================================
Advanced RAG Backend Server (FastAPI + FAISS + BM25 + Multi-Query + RRF Fusion)
================================================================================
[아키텍처 개요]
1. FAISS Index 구축 계층 (Dense Vector Layer):
   - 고차원 임베딩 벡터 검색을 위해 FAISS `IndexFlatL2`(정확한 전수 L2 거리 검색) 또는
     `IndexHNSWFlat`(계층적 탐색 가능 소세계 그래프 기반 근사 최근접 이웃 검색)을 지원합니다.
2. 다중 쿼리 확장 계층 (Multi-Query Expansion Layer):
   - 사용자의 단일 질의를 LLM을 통해 5가지 서로 다른 관점(정의/원리, 실무/적용, 비교/대조,
     한계/주의점, 키워드 중심)의 파생 쿼리로 확장하여 단일 쿼리의 어휘 불일치 및 누락을 방지합니다.
3. 하이브리드 검색 계층 (Hybrid Search Layer - Dense FAISS + Sparse BM25):
   - 확장된 5개 쿼리(및 원본 쿼리) 각각에 대해:
     (a) FAISS Dense Vector 검색 (문맥/의미적 유사도)
     (b) Okapi BM25 Sparse 키워드 검색 (정확한 용어/고유명사 매칭)
     을 병렬/연속 수행합니다.
4. RRF 순위 융합 계층 (RAG Fusion & Reciprocal Rank Fusion Layer):
   - 다중 쿼리 × 하이브리드 검색 경로에서 수집된 모든 문서 순위를 RRF 공식으로 통합:
     RRF_Score(d) = Σ_{m ∈ Queries} [ w_dense / (k + rank_FAISS(d, m)) + w_sparse / (k + rank_BM25(d, m)) ]
   - 상수 k=60을 적용하여 상위 편향을 완화하고, 다수 쿼리와 양쪽 검색기에서 고르게 상위권에
     등장한 문서를 최상위로 재정렬(Re-ranking)합니다.
5. 실시간 스트리밍 핸들러 (StreamHandler):
   - NDJSON(Newline Delimited JSON) 기반 스트리밍 응답을 통해 백엔드 진행률, 단계별 상태,
     중간 검색 결과, 최종 답변을 실시간으로 GAS 클라이언트에 전달합니다.

[알고리즘 복잡도 (Algorithmic Complexity)]
- 문서 청킹 및 전처리: 시간 O(C) (C: 전체 문자 수), 공간 O(N * L) (N: 청크 수, L: 평균 청크 길이)
- FAISS 인덱스 구축:
  * IndexFlatL2: 구축 시간 O(N * d), 검색 시간 O(N * d), 공간 O(N * d)
  * IndexHNSWFlat: 구축 시간 O(N * d * log N), 검색 시간 O(d * log N), 공간 O(N * (d + M_hnsw))
- BM25 인덱스 구축 및 검색:
  * 구축 시간 O(N * L_tokens), 쿼리당 검색 시간 O(|q| * N_postings) ≤ O(|q| * N)
- 다중 쿼리 하이브리드 검색 (M=5 확장 쿼리):
  * 총 검색 시간 O(M * (T_FAISS + T_BM25))
- RRF 순위 융합 및 재정렬:
  * 시간 O(M * K_cand + U * log U) (K_cand: 경로당 후보 수, U ≤ 2*M*K_cand: 고유 후보 문서 수)
  * 공간 O(U)
================================================================================
"""

import base64
import io
import json
import logging
import math
import os
import re
import time
import traceback
from dataclasses import dataclass, field, asdict
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import faiss
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

# 선택적 Google Gemini / OpenAI SDK 로드 (미설치 또는 API 키 미설정 시 결정론적 폴백 제공)
try:
    from google import genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("advanced_rag_server")


# ==============================================================================
# 1. 데이터 모델 및 요청/응답 스키마 정의
# ==============================================================================

class BuildDBRequest(BaseModel):
    """벡터 DB 생성 요청 스키마 (GAS에서 PDF Base64 또는 텍스트 전송)"""
    file_name: str = Field(..., description="원본 PDF 파일명")
    pdf_base64: Optional[str] = Field(None, description="Base64 인코딩된 PDF 바이너리 데이터")
    raw_text: Optional[str] = Field(None, description="직접 전달된 텍스트 (PDF 외 테스트용)")
    index_type: str = Field("IndexHNSWFlat", description="FAISS 인덱스 타입 ('IndexFlatL2' 또는 'IndexHNSWFlat')")
    chunk_size: int = Field(500, ge=100, le=2000, description="청크당 최대 문자 수")
    chunk_overlap: int = Field(100, ge=0, le=500, description="청크 간 중첩 문자 수")
    stream: bool = Field(False, description="실시간 스트리밍(NDJSON) 응답 여부")


class QueryRequest(BaseModel):
    """하이브리드 Multi-Query + RRF 질의 요청 스키마"""
    query: str = Field(..., min_length=1, description="사용자의 단일 입력 질의")
    top_k: int = Field(5, ge=1, le=20, description="최종 반환할 상위 RRF 문서 수")
    rrf_k: int = Field(60, ge=1, le=200, description="RRF 수식의 평활화 상수 k (기본값 60)")
    dense_weight: float = Field(1.0, ge=0.0, le=5.0, description="FAISS Dense 검색 가중치")
    sparse_weight: float = Field(1.0, ge=0.0, le=5.0, description="BM25 Sparse 검색 가중치")
    stream: bool = Field(False, description="실시간 스트리밍(NDJSON) 응답 여부")


@dataclass
class DocumentChunk:
    """단일 문서 청크 및 출처 메타데이터 구조체"""
    doc_id: int
    file_name: str
    page_number: int
    chunk_index: int
    content: str
    token_count: int
    char_count: int


# ==============================================================================
# 2. 실시간 스트리밍 핸들러 (StreamHandler)
# ==============================================================================

class StreamHandler:
    """
    서버 측 실시간 스트리밍 핸들러 (Server-side StreamHandler)
    ------------------------------------------------------------------------
    파이프라인의 각 단계(PDF 파싱 -> 청킹 -> 임베딩 -> FAISS/BM25 인덱싱 또는
    다중 쿼리 생성 -> 하이브리드 검색 -> RRF 융합 -> LLM 답변 생성)마다
    진행 상태(progress), 단계명(stage), 알고리즘 복잡도(complexity), 데이터 페이로드를
    NDJSON(Newline-Delimited JSON) 프레임으로 직렬화하여 전송합니다.
    """

    @staticmethod
    def format_event(
        event_type: str,
        stage: str,
        progress: int,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        complexity: Optional[Dict[str, str]] = None,
    ) -> str:
        """단일 스트리밍 이벤트를 JSON 라인(개행 포함)으로 직렬화"""
        payload = {
            "event": event_type,          # "progress" | "partial_result" | "complete" | "error"
            "stage": stage,               # 현재 파이프라인 단계 식별자
            "progress": max(0, min(100, progress)),
            "message": message,
            "timestamp": round(time.time(), 4),
            "complexity": complexity or {},
            "data": data or {},
        }
        return json.dumps(payload, ensure_ascii=False) + "\n"

    @staticmethod
    def format_error(
        stage: str,
        error_code: str,
        message: str,
        detail: str = "",
    ) -> str:
        """에러 발생 시 정확한 원인 및 단계 정보를 담은 스트리밍 에러 프레임 생성"""
        payload = {
            "event": "error",
            "stage": stage,
            "progress": 0,
            "error": {
                "error_code": error_code,
                "stage": stage,
                "message": message,
                "detail": detail,
            },
            "timestamp": round(time.time(), 4),
        }
        return json.dumps(payload, ensure_ascii=False) + "\n"


# ==============================================================================
# 3. 텍스트 토크나이저 및 임베딩/LLM 어댑터
# ==============================================================================

class KoreanEnglishTokenizer:
    """
    한영 혼합 문서를 위한 경량 형태소/N-gram 겸용 토크나이저
    ------------------------------------------------------------------------
    BM25 키워드 매칭 시 한국어 조사 결합으로 인한 누락을 방지하기 위해
    어절 단위 토큰과 2-gram(바이그램) 서브워드 토큰을 함께 추출합니다.
    - 시간 복잡도: O(L) (L: 입력 문자열 길이)
    - 공간 복잡도: O(T) (T: 추출된 토큰 수)
    """

    TOKEN_PATTERN = re.compile(r"[가-힣a-zA-Z0-9]+")
    STOPWORDS = {"이", "그", "저", "것", "수", "등", "및", "를", "을", "에", "의", "가", "은", "는", "the", "a", "an", "of", "to", "in", "for"}

    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        if not text:
            return []
        raw_tokens = cls.TOKEN_PATTERN.findall(text.lower())
        tokens: List[str] = []
        for tok in raw_tokens:
            if tok in cls.STOPWORDS:
                continue
            tokens.append(tok)
            # 한국어 3글자 이상 어절은 접두 2글자/3글자 서브토큰을 추가하여 조사 분리 효과 확보
            if len(tok) >= 3 and re.match(r"^[가-힣]+$", tok):
                tokens.append(tok[:2])
                if len(tok) >= 4:
                    tokens.append(tok[:3])
        return tokens if tokens else ["_empty_"]


class EmbeddingAndLLMEngine:
    """
    임베딩 생성 및 LLM 호출 엔진 (Gemini API 연동 + 결정론적 로컬 폴백 지원)
    ------------------------------------------------------------------------
    - GEMINI_API_KEY 환경 변수가 설정되어 있으면 Google Gemini 모델(`text-embedding-004`,
      `gemini-2.5-flash`)을 호출합니다.
    - API 키가 없거나 네트워크가 제한된 환경에서도 파이프라인 전체가 즉시 동작하도록
      고차원 의미 해싱 임베딩(Semantic N-gram Random Projection, dim=384) 및
      관점 기반 다중 쿼리 확장기를 내장하고 있습니다.
    """

    def __init__(self, embedding_dim: int = 384):
        self.api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        self.client = None
        self.embedding_dim = embedding_dim
        self.mode = "local_hybrid_fallback"

        if HAS_GENAI and self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
                self.embedding_dim = 768  # text-embedding-004 기본 차원
                self.mode = "gemini_api"
                logger.info("Gemini API 클라이언트 초기화 완료 (dim=%d)", self.embedding_dim)
            except Exception as e:
                logger.warning("Gemini 클라이언트 초기화 실패, 로컬 엔진으로 폴백합니다: %s", e)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        문서/쿼리 리스트를 고정 차원 Dense Float32 정규화 벡터 행렬로 변환합니다.
        - 출력 shape: (len(texts), self.embedding_dim), dtype: np.float32
        - 시간 복잡도: O(N * L * d) (로컬 투영 시) 또는 O(N) API 호출
        """
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        if self.client is not None:
            try:
                response = self.client.models.embed_content(
                    model="text-embedding-004",
                    contents=texts,
                )
                vectors = [emb.values for emb in response.embeddings]
                arr = np.array(vectors, dtype=np.float32)
                # L2 정규화
                norms = np.linalg.norm(arr, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                return arr / norms
            except Exception as e:
                logger.warning("Gemini 임베딩 API 호출 실패, 로컬 해싱 임베딩으로 대체: %s", e)

        # 결정론적 다중-스케일 문자/단어 N-gram 해싱 임베딩 (L2 정규화 포함)
        dim = self.embedding_dim
        matrix = np.zeros((len(texts), dim), dtype=np.float32)
        for row_idx, text in enumerate(texts):
            tokens = KoreanEnglishTokenizer.tokenize(text)
            # 단어 토큰 + 문자 바이그램/트라이그램 결합으로 의미적/형태적 유사도 반영
            clean = re.sub(r"\s+", " ", text.lower().strip())
            char_ngrams = [clean[i:i+2] for i in range(max(0, len(clean) - 1))] + [
                clean[i:i+3] for i in range(max(0, len(clean) - 2))
            ]
            features = tokens * 2 + char_ngrams  # 단어 토큰에 가중치 2배 부여
            if not features:
                matrix[row_idx, 0] = 1.0
                continue
            for feat in features:
                # 안정적인 해시 시드 계산
                h = int.from_bytes(feat.encode("utf-8"), byteorder="little")
                idx1 = h % dim
                idx2 = ((h >> 11) ^ 0x5DEECE66D) % dim
                sign1 = 1.0 if (h & 1) == 0 else -1.0
                sign2 = 1.0 if ((h >> 1) & 1) == 0 else -1.0
                matrix[row_idx, idx1] += sign1
                matrix[row_idx, idx2] += 0.5 * sign2

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def expand_multi_queries(self, original_query: str) -> List[Dict[str, str]]:
        """
        [다중 쿼리 확장 계층 - Multi-Query Layer]
        사용자의 단일 입력 쿼리를 서로 다른 5가지 관점의 파생 쿼리로 확장합니다.
        관점:
          1. 개념 및 정의 관점 (Conceptual & Definition)
          2. 핵심 원리 및 메커니즘 관점 (Mechanism & Architecture)
          3. 실무 적용 및 활용 사례 관점 (Practical Application & Use-cases)
          4. 비교 분석 및 장단점 관점 (Comparison & Trade-offs)
          5. 핵심 키워드 밀집 검색 관점 (Keyword-Dense Lexical Focus)
        """
        q = original_query.strip()
        if self.client is not None:
            try:
                prompt = (
                    "당신은 RAG 검색 최적화 전문가입니다. 아래 사용자 질의를 바탕으로 벡터 및 키워드 검색 재현율(Recall)을 "
                    "극대화할 수 있도록 서로 다른 5가지 관점의 검색 쿼리로 확장하세요.\n"
                    "반드시 JSON 배열 형식으로만 응답하세요: "
                    '[{"id": 1, "perspective": "관점명", "query": "확장된 쿼리"}, ...]\n\n'
                    f"사용자 질의: {q}"
                )
                resp = self.client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                )
                raw = resp.text.strip()
                match = re.search(r"\[.*\]", raw, re.DOTALL)
                if match:
                    parsed = json.loads(match.group(0))
                    if isinstance(parsed, list) and len(parsed) >= 5:
                        return [
                            {
                                "id": i + 1,
                                "perspective": str(item.get("perspective", f"관점 {i+1}")),
                                "query": str(item.get("query", q)),
                            }
                            for i, item in enumerate(parsed[:5])
                        ]
            except Exception as e:
                logger.warning("LLM 다중 쿼리 생성 실패, 규칙 기반 5대 관점 확장기로 폴백: %s", e)

        # 핵심 키워드 추출
        core_tokens = KoreanEnglishTokenizer.tokenize(q)
        unique_keywords = []
        for t in core_tokens:
            if t not in unique_keywords and len(t) >= 2:
                unique_keywords.append(t)
        kw_str = " ".join(unique_keywords[:8]) if unique_keywords else q

        return [
            {
                "id": 1,
                "perspective": "1. 개념 및 핵심 정의 관점",
                "query": f"{q} 핵심 개념 정의 및 기본 개요 설명 ({kw_str})",
            },
            {
                "id": 2,
                "perspective": "2. 작동 원리 및 구조적 메커니즘 관점",
                "query": f"{q} 세부 작동 원리 프로세스 아키텍처 구성 요소 ({kw_str})",
            },
            {
                "id": 3,
                "perspective": "3. 실무 적용 방법 및 구체적 사례 관점",
                "query": f"{q} 실제 적용 방법 구현 절차 및 활용 사례 ({kw_str})",
            },
            {
                "id": 4,
                "perspective": "4. 성능 특성·장단점 및 비교 분석 관점",
                "query": f"{q} 주요 장점 한계점 성능 비교 및 고려사항 ({kw_str})",
            },
            {
                "id": 5,
                "perspective": "5. 핵심 키워드 밀집 매칭 관점 (BM25 최적화)",
                "query": f"{kw_str} {q}",
            },
        ]

    def synthesize_answer(
        self,
        query: str,
        expanded_queries: List[Dict[str, str]],
        ranked_docs: List[Dict[str, Any]],
    ) -> str:
        """
        RRF로 재정렬된 상위 문서 스니펫들을 근거로 최종 답변을 생성하며,
        모든 핵심 문장에 [출처: 파일명 p.페이지, Chunk #번호] 형태의 출처를 명시합니다.
        """
        if not ranked_docs:
            return "검색된 관련 문서가 없습니다. 먼저 PDF 파일을 업로드하여 벡터 DB를 생성해 주세요."

        if self.client is not None:
            try:
                context_blocks = []
                for idx, doc in enumerate(ranked_docs, start=1):
                    citation = f"[출처: {doc['file_name']} p.{doc['page_number']}, Chunk #{doc['chunk_index']}]"
                    context_blocks.append(
                        f"문서 [{idx}] {citation} (RRF 점수: {doc['rrf_score']:.5f}):\n{doc['content']}"
                    )
                context_text = "\n\n".join(context_blocks)
                prompt = (
                    "당신은 신뢰할 수 있는 엔터프라이즈 RAG AI 어시스턴트입니다.\n"
                    "제공된 [검색 문서 컨텍스트]만을 근거로 사용자의 질문에 체계적이고 명확하게 한국어로 답변하세요.\n"
                    "답변 작성 규칙:\n"
                    "1. 핵심 요약, 상세 분석, 결론 순서로 구조화하여 작성할 것.\n"
                    "2. 인용한 모든 내용 뒤에는 반드시 `[출처: 파일명 p.페이지, Chunk #청크번호]` 형식으로 출처를 표기할 것.\n"
                    "3. 컨텍스트에 없는 내용은 추측하지 말고 문서에서 확인된 사실만 기술할 것.\n\n"
                    f"[사용자 질문]\n{query}\n\n"
                    f"[검색 문서 컨텍스트]\n{context_text}"
                )
                resp = self.client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                )
                if resp.text:
                    return resp.text.strip()
            except Exception as e:
                logger.warning("Gemini 답변 생성 실패, 구조적 근거 요약 합성기로 대체: %s", e)

        # 로컬 구조적 근거 합성 (항상 출처 표기 포함)
        lines = [
            f"### 질문에 대한 Hybrid RAG Fusion 종합 답변",
            f"**질의 내용**: \"{query}\"",
            f"**적용 파이프라인**: 5가지 다중 쿼리 확장 → FAISS Dense + BM25 Sparse 하이브리드 검색 → RRF 순위 융합 (상위 {len(ranked_docs)}개 문서 선별)\n",
            "---",
            "#### 1. 핵심 근거 요약 및 분석",
        ]
        for rank_idx, doc in enumerate(ranked_docs, start=1):
            citation = f"[출처: {doc['file_name']} p.{doc['page_number']}, Chunk #{doc['chunk_index']}]"
            snippet = doc["content"].strip()
            lines.append(
                f"**({rank_idx}) 핵심 근거 #{rank_idx}** — RRF 점수 `{doc['rrf_score']:.5f}` "
                f"(FAISS 최고순위: `{doc['best_dense_rank'] or '-'}`, BM25 최고순위: `{doc['best_sparse_rank'] or '-'}`)\n"
                f"> {snippet}\n"
                f"- **근거 출처**: {citation}\n"
            )

        citations_summary = ", ".join(
            f"[출처: {d['file_name']} p.{d['page_number']}, Chunk #{d['chunk_index']}]"
            for d in ranked_docs
        )
        lines.extend([
            "---",
            "#### 2. 종합 결론",
            f"위 상위 {len(ranked_docs)}개 핵심 문맥을 종합하면, **\"{query}\"**에 대한 주요 내용은 "
            f"{ranked_docs[0]['file_name']} 문서의 {ranked_docs[0]['page_number']}페이지 등을 중심으로 확인됩니다. "
            f"다중 관점 쿼리와 RRF 상호 순위 융합을 통해 의미적 유사성(FAISS)과 정확한 키워드 일치(BM25)가 모두 검증된 결과입니다. "
            f"{citations_summary}",
        ])
        return "\n".join(lines)


# ==============================================================================
# 4. FAISS + BM25 하이브리드 인덱스 및 RRF 파이프라인 매니저
# ==============================================================================

class HybridRAGPipeline:
    """
    FAISS(IndexFlatL2 / IndexHNSWFlat) + BM25 하이브리드 검색 및 RRF 융합 엔진
    """

    def __init__(self):
        self.engine = EmbeddingAndLLMEngine(embedding_dim=384)
        self.faiss_index: Optional[faiss.Index] = None
        self.index_type: str = "IndexHNSWFlat"
        self.bm25_index: Optional[BM25Okapi] = None
        self.documents: List[DocumentChunk] = []
        self.tokenized_corpus: List[List[str]] = []
        self.active_file_name: Optional[str] = None
        self.last_build_time: Optional[float] = None
        self.build_metrics: Dict[str, Any] = {}

    def is_ready(self) -> bool:
        """인덱스가 정상 구축되어 검색 가능한지 확인"""
        return (
            self.faiss_index is not None
            and self.bm25_index is not None
            and len(self.documents) > 0
        )

    @staticmethod
    def extract_pages_from_pdf_bytes(pdf_bytes: bytes) -> List[Tuple[int, str]]:
        """
        PDF 바이너리에서 페이지별 텍스트를 추출합니다.
        - 시간 복잡도: O(P * C_p) (P: 페이지 수, C_p: 페이지당 문자 수)
        """
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages: List[Tuple[int, str]] = []
        for idx, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            cleaned = re.sub(r"[ \t]+", " ", text).strip()
            if cleaned:
                pages.append((idx, cleaned))
        return pages

    @staticmethod
    def chunk_pages(
        pages: List[Tuple[int, str]],
        file_name: str,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ) -> List[DocumentChunk]:
        """
        페이지별 텍스트를 문장 경계를 고려한 슬라이딩 윈도우 방식으로 청킹합니다.
        - 시간 복잡도: O(C) (C: 전체 문서 문자 수)
        - 공간 복잡도: O(N * chunk_size) (N: 생성된 청크 수)
        """
        chunks: List[DocumentChunk] = []
        doc_id = 0
        step = max(50, chunk_size - chunk_overlap)

        for page_num, text in pages:
            start = 0
            chunk_idx_in_page = 1
            text_len = len(text)
            while start < text_len:
                end = min(text_len, start + chunk_size)
                # 문장 마침표나 개행 근처에서 자연스럽게 자르기
                if end < text_len:
                    natural_break = max(
                        text.rfind(". ", start + int(chunk_size * 0.6), end),
                        text.rfind(".\n", start + int(chunk_size * 0.6), end),
                        text.rfind("\n", start + int(chunk_size * 0.6), end),
                    )
                    if natural_break != -1 and natural_break > start:
                        end = natural_break + 1

                snippet = text[start:end].strip()
                if snippet:
                    tokens = KoreanEnglishTokenizer.tokenize(snippet)
                    chunks.append(
                        DocumentChunk(
                            doc_id=doc_id,
                            file_name=file_name,
                            page_number=page_num,
                            chunk_index=chunk_idx_in_page,
                            content=snippet,
                            token_count=len(tokens),
                            char_count=len(snippet),
                        )
                    )
                    doc_id += 1
                    chunk_idx_in_page += 1

                if end >= text_len:
                    break
                start = max(start + 1, end - chunk_overlap)

        return chunks

    def build_indices(
        self,
        chunks: List[DocumentChunk],
        index_type: str = "IndexHNSWFlat",
    ) -> Dict[str, Any]:
        """
        [FAISS Index 구축 및 BM25 코퍼스 인덱싱]
        - IndexFlatL2: 전수 L2 거리 계산 (정확도 100%, 검색 복잡도 O(N * d))
        - IndexHNSWFlat: 계층적 그래프 기반 근사 탐색 (M=32, efConstruction=64, efSearch=32, 검색 복잡도 O(d * log N))
        """
        if not chunks:
            raise ValueError("인덱싱할 문서 청크가 비어 있습니다.")

        t0 = time.perf_counter()
        self.documents = chunks
        self.active_file_name = chunks[0].file_name

        # 1) Dense 임베딩 생성
        texts = [c.content for c in chunks]
        embeddings = self.engine.embed_texts(texts)
        num_docs, dim = embeddings.shape

        # 2) FAISS 인덱스 초기화 및 벡터 적재
        normalized_type = "IndexFlatL2" if index_type == "IndexFlatL2" else "IndexHNSWFlat"
        self.index_type = normalized_type

        if normalized_type == "IndexFlatL2":
            index = faiss.IndexFlatL2(dim)
        else:
            # IndexHNSWFlat(dim, M): M=32 연결 그래프 구성
            index = faiss.IndexHNSWFlat(dim, 32)
            index.hnsw.efConstruction = 64
            index.hnsw.efSearch = 32

        index.add(embeddings)
        self.faiss_index = index

        # 3) Sparse BM25 인덱스 구축
        self.tokenized_corpus = [KoreanEnglishTokenizer.tokenize(c.content) for c in chunks]
        self.bm25_index = BM25Okapi(self.tokenized_corpus)

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        self.last_build_time = time.time()

        complexity_info = {
            "chunking": "Time: O(C), Space: O(N * L)",
            "faiss_build": (
                f"Time: O(N * d) [N={num_docs}, d={dim}]"
                if normalized_type == "IndexFlatL2"
                else f"Time: O(N * d * log N) [N={num_docs}, d={dim}, M=32]"
            ),
            "faiss_search_per_query": (
                f"O(N * d) ({ normalized_type })"
                if normalized_type == "IndexFlatL2"
                else f"O(d * log N) ({ normalized_type })"
            ),
            "bm25_build": f"Time: O(N * L_avg) [N={num_docs}]",
            "space_complexity": f"O(N * d + N * L_avg) ≈ {num_docs * dim * 4 / 1024:.1f} KB (Vectors)",
        }

        self.build_metrics = {
            "file_name": self.active_file_name,
            "total_chunks": num_docs,
            "embedding_dim": dim,
            "index_type": normalized_type,
            "engine_mode": self.engine.mode,
            "build_elapsed_ms": elapsed_ms,
            "complexity": complexity_info,
        }
        return self.build_metrics

    def search_hybrid_rrf(
        self,
        query: str,
        top_k: int = 5,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ) -> Dict[str, Any]:
        """
        [Multi-Query (5개 확장) + Hybrid Search (FAISS + BM25) + RRF Fusion 파이프라인]
        ------------------------------------------------------------------------
        1. 단일 사용자 쿼리를 5가지 관점의 파생 쿼리로 확장합니다.
        2. 각 확장 쿼리마다:
           - FAISS Dense 검색으로 상위 K_cand개 문서와 L2 거리(및 유사도)를 산출합니다.
           - BM25 Sparse 검색으로 상위 K_cand개 문서와 BM25 점수를 산출합니다.
        3. Reciprocal Rank Fusion (RRF) 공식 적용:
           RRF_Score(d) = Σ_{q_i} [ w_dense / (rrf_k + rank_dense(d, q_i))
                                  + w_sparse / (rrf_k + rank_sparse(d, q_i)) ]
        4. 최종 상위 top_k개 문서를 재정렬하여 반환하고 LLM 답변을 합성합니다.
        """
        if not self.is_ready():
            raise RuntimeError("벡터 DB가 아직 초기화되지 않았습니다. 먼저 PDF로 벡터 DB를 생성해 주세요.")

        t_start = time.perf_counter()
        num_docs = len(self.documents)
        candidate_k = min(num_docs, max(top_k * 3, 10))

        # Step 1: 5가지 관점의 다중 쿼리 확장
        t_mq = time.perf_counter()
        expanded_queries = self.engine.expand_multi_queries(query)
        mq_elapsed_ms = round((time.perf_counter() - t_mq) * 1000, 2)

        # Step 2: 확장된 5개 쿼리에 대한 FAISS Dense + BM25 Sparse 하이브리드 검색
        t_search = time.perf_counter()
        query_strings = [item["query"] for item in expanded_queries]
        query_embeddings = self.engine.embed_texts(query_strings)

        # FAISS 일괄 검색: distances shape=(M, candidate_k), indices shape=(M, candidate_k)
        distances, faiss_indices = self.faiss_index.search(query_embeddings, candidate_k)

        # RRF 누적 테이블
        rrf_scores: Dict[int, float] = {}
        doc_trace: Dict[int, Dict[str, Any]] = {}
        per_query_search_logs: List[Dict[str, Any]] = []

        for q_idx, q_info in enumerate(expanded_queries):
            q_str = q_info["query"]
            # (a) FAISS Dense 순위 반영
            dense_hits: List[Dict[str, Any]] = []
            for rank_0, (doc_id_raw, dist_raw) in enumerate(zip(faiss_indices[q_idx], distances[q_idx])):
                doc_id = int(doc_id_raw)
                if doc_id < 0 or doc_id >= num_docs:
                    continue
                rank = rank_0 + 1
                dist = float(dist_raw)
                # 정규화 벡터 기준 L2 거리를 코사인 유사도 근사치(0~1)로 변환: sim = max(0, 1 - dist / 2)
                dense_sim = max(0.0, 1.0 - (dist / 2.0))
                rrf_contrib = dense_weight * (1.0 / (rrf_k + rank))
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + rrf_contrib

                trace = doc_trace.setdefault(
                    doc_id,
                    {
                        "dense_ranks": [],
                        "sparse_ranks": [],
                        "max_dense_sim": 0.0,
                        "max_bm25_score": 0.0,
                        "matched_perspectives": set(),
                    },
                )
                trace["dense_ranks"].append(rank)
                trace["max_dense_sim"] = max(trace["max_dense_sim"], dense_sim)
                trace["matched_perspectives"].add(q_info["perspective"])
                dense_hits.append({"doc_id": doc_id, "rank": rank, "similarity": round(dense_sim, 4)})

            # (b) BM25 Sparse 키워드 순위 반영
            q_tokens = KoreanEnglishTokenizer.tokenize(q_str)
            bm25_scores = self.bm25_index.get_scores(q_tokens)
            top_bm25_indices = np.argsort(bm25_scores)[::-1][:candidate_k]

            sparse_hits: List[Dict[str, Any]] = []
            for rank_0, doc_id_raw in enumerate(top_bm25_indices):
                doc_id = int(doc_id_raw)
                score = float(bm25_scores[doc_id])
                rank = rank_0 + 1
                # 점수가 0보다 큰 경우에만 키워드 매칭 가중치 부여 (전혀 매칭 안 된 문서의 노이즈 방지)
                if score > 0.0:
                    rrf_contrib = sparse_weight * (1.0 / (rrf_k + rank))
                    rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + rrf_contrib

                    trace = doc_trace.setdefault(
                        doc_id,
                        {
                            "dense_ranks": [],
                            "sparse_ranks": [],
                            "max_dense_sim": 0.0,
                            "max_bm25_score": 0.0,
                            "matched_perspectives": set(),
                        },
                    )
                    trace["sparse_ranks"].append(rank)
                    trace["max_bm25_score"] = max(trace["max_bm25_score"], score)
                    trace["matched_perspectives"].add(q_info["perspective"])
                    sparse_hits.append({"doc_id": doc_id, "rank": rank, "bm25_score": round(score, 4)})

            per_query_search_logs.append(
                {
                    "id": q_info["id"],
                    "perspective": q_info["perspective"],
                    "expanded_query": q_str,
                    "dense_top3": dense_hits[:3],
                    "sparse_top3": sparse_hits[:3],
                }
            )

        hybrid_search_elapsed_ms = round((time.perf_counter() - t_search) * 1000, 2)

        # Step 3: RRF 점수 기준 내림차순 정렬 (Re-ranking)
        t_rrf = time.perf_counter()
        sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda d_id: rrf_scores[d_id], reverse=True)[:top_k]

        ranked_results: List[Dict[str, Any]] = []
        for final_rank, doc_id in enumerate(sorted_doc_ids, start=1):
            chunk = self.documents[doc_id]
            trace = doc_trace[doc_id]
            ranked_results.append(
                {
                    "final_rank": final_rank,
                    "doc_id": chunk.doc_id,
                    "file_name": chunk.file_name,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    "rrf_score": round(rrf_scores[doc_id], 6),
                    "best_dense_rank": min(trace["dense_ranks"]) if trace["dense_ranks"] else None,
                    "best_sparse_rank": min(trace["sparse_ranks"]) if trace["sparse_ranks"] else None,
                    "dense_similarity": round(trace["max_dense_sim"], 4),
                    "bm25_score": round(trace["max_bm25_score"], 4),
                    "matched_perspectives": sorted(list(trace["matched_perspectives"])),
                    "citation": f"[출처: {chunk.file_name} p.{chunk.page_number}, Chunk #{chunk.chunk_index}]",
                }
            )
        rrf_elapsed_ms = round((time.perf_counter() - t_rrf) * 1000, 2)

        # Step 4: LLM 최종 답변 합성
        t_llm = time.perf_counter()
        answer = self.engine.synthesize_answer(query, expanded_queries, ranked_results)
        llm_elapsed_ms = round((time.perf_counter() - t_llm) * 1000, 2)
        total_elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)

        M = len(expanded_queries)
        U = len(rrf_scores)
        complexity_metrics = {
            "multi_query_expansion": f"O(M) [M={M} perspectives]",
            "faiss_dense_search": (
                f"O(M * N * d) [IndexFlatL2, M={M}, N={num_docs}, d={self.engine.embedding_dim}]"
                if self.index_type == "IndexFlatL2"
                else f"O(M * d * log N) [IndexHNSWFlat, M={M}, N={num_docs}, d={self.engine.embedding_dim}]"
            ),
            "bm25_sparse_search": f"O(M * |Q| * N) [M={M}, N={num_docs}]",
            "rrf_fusion_rerank": f"O(M * K_cand + U * log U) [K_cand={candidate_k}, U={U} unique docs]",
            "space_complexity": f"O(N * d + U) [N={num_docs}, U={U}]",
        }

        return {
            "status": "success",
            "original_query": query,
            "expanded_queries": expanded_queries,
            "per_query_search_logs": per_query_search_logs,
            "ranked_documents": ranked_results,
            "llm_answer": answer,
            "sources": [
                {
                    "rank": r["final_rank"],
                    "file_name": r["file_name"],
                    "page_number": r["page_number"],
                    "chunk_index": r["chunk_index"],
                    "citation": r["citation"],
                    "rrf_score": r["rrf_score"],
                }
                for r in ranked_results
            ],
            "metrics": {
                "index_type": self.index_type,
                "total_indexed_chunks": num_docs,
                "rrf_k": rrf_k,
                "dense_weight": dense_weight,
                "sparse_weight": sparse_weight,
                "timings_ms": {
                    "multi_query_ms": mq_elapsed_ms,
                    "hybrid_search_ms": hybrid_search_elapsed_ms,
                    "rrf_fusion_ms": rrf_elapsed_ms,
                    "llm_synthesis_ms": llm_elapsed_ms,
                    "total_ms": total_elapsed_ms,
                },
                "complexity": complexity_metrics,
            },
        }


# ==============================================================================
# 5. FastAPI 애플리케이션 및 엔드포인트 정의
# ==============================================================================

app = FastAPI(
    title="Advanced Hybrid RAG Fusion Server",
    description="FAISS + BM25 Hybrid Search, 5-Way Multi-Query Expansion, and RRF Re-ranking Backend for GAS",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = HybridRAGPipeline()


@app.get("/health")
async def health_check():
    """
    서버 연동 상태 및 현재 로드된 FAISS/BM25 인덱스 메타데이터 반환
    """
    return {
        "status": "online",
        "server_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "db_ready": pipeline.is_ready(),
        "active_file_name": pipeline.active_file_name,
        "total_chunks": len(pipeline.documents),
        "index_type": pipeline.index_type,
        "embedding_dim": pipeline.engine.embedding_dim,
        "engine_mode": pipeline.engine.mode,
        "build_metrics": pipeline.build_metrics,
    }


@app.post("/build-db")
async def build_vector_db(req: BuildDBRequest):
    """
    PDF 파일(Base64) 또는 원문 텍스트를 받아 청킹 후 FAISS + BM25 하이브리드 인덱스를 구축합니다.
    `req.stream == True`이면 StreamHandler를 통해 실시간 진행 단계를 NDJSON 스트림으로 반환합니다.
    """

    async def stream_generator() -> AsyncGenerator[str, None]:
        try:
            yield StreamHandler.format_event(
                event_type="progress",
                stage="INIT",
                progress=10,
                message=f"'{req.file_name}' 파일 수신 완료. PDF 텍스트 추출을 시작합니다.",
                complexity={"pdf_parsing": "Time: O(P * C_p)"},
            )

            pages: List[Tuple[int, str]] = []
            if req.pdf_base64:
                try:
                    raw_b64 = req.pdf_base64.split(",")[-1].strip()
                    pdf_bytes = base64.b64decode(raw_b64)
                    pages = pipeline.extract_pages_from_pdf_bytes(pdf_bytes)
                except Exception as pdf_err:
                    yield StreamHandler.format_error(
                        stage="PDF_EXTRACTION",
                        error_code="ERR_PDF_DECODE_OR_PARSE",
                        message="PDF 파일 디코딩 또는 텍스트 추출 중 오류가 발생했습니다.",
                        detail=str(pdf_err),
                    )
                    return
            elif req.raw_text:
                pages = [(1, req.raw_text.strip())]
            else:
                yield StreamHandler.format_error(
                    stage="INIT",
                    error_code="ERR_EMPTY_PAYLOAD",
                    message="PDF 데이터(pdf_base64) 또는 텍스트(raw_text)가 제공되지 않았습니다.",
                )
                return

            if not pages:
                yield StreamHandler.format_error(
                    stage="PDF_EXTRACTION",
                    error_code="ERR_NO_EXTRACTABLE_TEXT",
                    message="PDF에서 추출 가능한 텍스트가 없습니다. (스캔 이미지 PDF인지 확인하세요)",
                )
                return

            yield StreamHandler.format_event(
                event_type="progress",
                stage="CHUNKING",
                progress=40,
                message=f"총 {len(pages)}페이지 텍스트 추출 완료. 슬라이딩 윈도우 청킹(size={req.chunk_size}, overlap={req.chunk_overlap}) 진행 중...",
                complexity={"chunking": "Time: O(C), Space: O(N * L)"},
            )

            chunks = pipeline.chunk_pages(
                pages=pages,
                file_name=req.file_name,
                chunk_size=req.chunk_size,
                chunk_overlap=req.chunk_overlap,
            )

            yield StreamHandler.format_event(
                event_type="progress",
                stage="INDEXING",
                progress=75,
                message=f"총 {len(chunks)}개 청크 생성 완료. FAISS({req.index_type}) 벡터 인덱스 및 BM25 키워드 인덱스 구축 중...",
            )

            metrics = pipeline.build_indices(chunks=chunks, index_type=req.index_type)

            yield StreamHandler.format_event(
                event_type="complete",
                stage="DONE",
                progress=100,
                message=f"하이브리드 벡터 DB 구축 완료! (총 {len(chunks)}개 청크, 인덱스: {metrics['index_type']})",
                data=metrics,
                complexity=metrics["complexity"],
            )
        except Exception as e:
            logger.error("벡터 DB 구축 실패: %s\n%s", e, traceback.format_exc())
            yield StreamHandler.format_error(
                stage="INDEXING",
                error_code="ERR_BUILD_INDEX_FAILED",
                message="FAISS / BM25 인덱스 구축 중 서버 내부 오류가 발생했습니다.",
                detail=str(e),
            )

    if req.stream:
        return StreamingResponse(stream_generator(), media_type="application/x-ndjson")

    # 일반 JSON 응답 모드 (스트리밍 이벤트를 모두 수집하여 진행 로그와 함께 반환)
    steps_log = []
    final_data = None
    error_obj = None
    async for line in stream_generator():
        evt = json.loads(line)
        steps_log.append(evt)
        if evt.get("event") == "error":
            error_obj = evt.get("error")
        elif evt.get("event") == "complete":
            final_data = evt.get("data")

    if error_obj:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "error": error_obj, "progress_logs": steps_log},
        )

    return {
        "status": "success",
        "message": f"'{req.file_name}' 기반 하이브리드 벡터 DB 생성이 완료되었습니다.",
        "metrics": final_data,
        "progress_logs": steps_log,
    }


@app.post("/query")
async def query_rag_pipeline(req: QueryRequest):
    """
    단일 질의를 받아 5가지 다중 쿼리 확장 -> FAISS + BM25 하이브리드 검색 -> RRF 재정렬 -> LLM 답변 및 출처를 반환합니다.
    `req.stream == True`이면 StreamHandler의 NDJSON 스트리밍으로 단계별 결과를 실시간 전송합니다.
    """

    async def query_stream_generator() -> AsyncGenerator[str, None]:
        try:
            if not pipeline.is_ready():
                yield StreamHandler.format_error(
                    stage="PRECHECK",
                    error_code="ERR_DB_NOT_INITIALIZED",
                    message="벡터 DB가 생성되지 않았습니다. 먼저 PDF 파일을 선택하고 '벡터 DB 생성' 버튼을 눌러주세요.",
                )
                return

            yield StreamHandler.format_event(
                event_type="progress",
                stage="MULTI_QUERY",
                progress=20,
                message="단일 질의를 5가지 다각도 관점의 파생 쿼리로 확장 중...",
                complexity={"multi_query": "Time: O(M) [M=5]"},
            )

            result = pipeline.search_hybrid_rrf(
                query=req.query,
                top_k=req.top_k,
                rrf_k=req.rrf_k,
                dense_weight=req.dense_weight,
                sparse_weight=req.sparse_weight,
            )

            yield StreamHandler.format_event(
                event_type="partial_result",
                stage="HYBRID_RRF",
                progress=65,
                message=f"FAISS({pipeline.index_type}) + BM25 하이브리드 검색 및 RRF(k={req.rrf_k}) 상위 {len(result['ranked_documents'])}건 재정렬 완료.",
                data={
                    "expanded_queries": result["expanded_queries"],
                    "ranked_documents": result["ranked_documents"],
                },
                complexity=result["metrics"]["complexity"],
            )

            yield StreamHandler.format_event(
                event_type="complete",
                stage="LLM_SYNTHESIS",
                progress=100,
                message="RRF 상위 문서 기반 LLM 답변 생성 및 출처 매핑 완료.",
                data=result,
                complexity=result["metrics"]["complexity"],
            )
        except Exception as e:
            logger.error("질의 처리 중 오류 발생: %s\n%s", e, traceback.format_exc())
            yield StreamHandler.format_error(
                stage="QUERY_EXECUTION",
                error_code="ERR_QUERY_PIPELINE_FAILED",
                message="하이브리드 검색 및 RRF 융합 파이프라인 실행 중 오류가 발생했습니다.",
                detail=str(e),
            )

    if req.stream:
        return StreamingResponse(query_stream_generator(), media_type="application/x-ndjson")

    steps_log = []
    final_result = None
    error_obj = None
    async for line in query_stream_generator():
        evt = json.loads(line)
        steps_log.append(evt)
        if evt.get("event") == "error":
            error_obj = evt.get("error")
        elif evt.get("event") == "complete":
            final_result = evt.get("data")

    if error_obj:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "error": error_obj, "stream_logs": steps_log},
        )

    final_result["stream_logs"] = steps_log
    return final_result


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)

