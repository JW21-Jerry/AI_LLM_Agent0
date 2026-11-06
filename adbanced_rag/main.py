<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Advanced Hybrid RAG (FAISS + BM25 + Multi-Query + RRF)</title>
  <!--
    ==========================================================================
    Advanced Hybrid RAG 클라이언트 UI (index.html)
    ==========================================================================
    [주요 UI 구성 및 기능]
    1. 서버와의 연동 상태 표시 영역:
       - Cloud Run URL 환경변수 표시/변경, 연결 상태 배지(Online/Offline),
         현재 활성 인덱스(IndexHNSWFlat / IndexFlatL2), 적재된 청크 수, 응답 지연시간 표시.
    2. 알고리즘 시간/공간 복잡도 상시 및 동적 표시 패널:
       - FAISS, BM25, Multi-Query(5가지), RRF 순위 융합의 이론적·실측 복잡도 표시.
    3. 파일명 표시 및 파일 요청 버튼 + 벡터 DB 생성 버튼:
       - Google Drive PDF 파일 요청 또는 로컬 PDF 파일 선택 지원.
       - 선택된 PDF 파일명 및 용량 표시, FAISS 인덱스 방식 선택 후 벡터 DB 생성 요청.
    4. 서버에서 RAG 생성 진행 현황 표시 영역:
       - StreamHandler가 수신한 실시간 단계별 진행률 바(Progress Bar) 및 타임라인 로그 렌더링.
    5. 질의 입력 영역 및 질의하기 버튼:
       - 사용자의 자연어 질의 입력 및 RRF 하이퍼파라미터(Top-K, RRF k) 설정.
    6. 답변 표시 영역 (분리형 레이아웃):
       - [영역 A] 5가지 관점으로 확장된 Multi-Query 목록
       - [영역 B] RAG 검색 내용: RRF로 재정렬된 상위 문서 스니펫, RRF 점수, FAISS/BM25 개별 순위, 메타데이터
       - [영역 C] LLM 최종 답변 및 명확한 출처 표기 ([출처: 파일명 p.X, Chunk #Y])
    7. 정확한 에러 표시 배너:
       - 에러 코드(error_code), 발생 단계(stage), 원인 메시지, 상세 디버그 정보를 직관적으로 표시.
    ==========================================================================
  -->
  <style>
    :root {
      --bg-primary: #0f172a;
      --bg-card: #1e293b;
      --bg-input: #0b1120;
      --border-color: #334155;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --accent-blue: #38bdf8;
      --accent-indigo: #6366f1;
      --accent-emerald: #10b981;
      --accent-amber: #f59e0b;
      --accent-rose: #f43f5e;
      --font-sans: 'Pretendard', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      --font-mono: 'JetBrains Mono', Consolas, monospace;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      font-family: var(--font-sans);
      background-color: var(--bg-primary);
      color: var(--text-main);
      line-height: 1.6;
      padding: 24px;
    }

    .container {
      max-width: 1280px;
      margin: 0 auto;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }

    /* 상단 헤더 및 서버 연동 상태 바 */
    .header-bar {
      background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      padding: 20px 24px;
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
    }

    .title-group h1 {
      font-size: 1.4rem;
      font-weight: 700;
      color: var(--text-main);
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .title-group p {
      font-size: 0.88rem;
      color: var(--text-muted);
      margin-top: 4px;
    }

    .status-cluster {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 12px;
    }

    .status-badge {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 14px;
      border-radius: 9999px;
      font-size: 0.82rem;
      font-weight: 600;
      border: 1px solid var(--border-color);
      background: var(--bg-input);
    }

    .status-dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background-color: var(--accent-amber);
    }

    .status-dot.online {
      background-color: var(--accent-emerald);
      box-shadow: 0 0 8px var(--accent-emerald);
    }

    .status-dot.offline {
      background-color: var(--accent-rose);
      box-shadow: 0 0 8px var(--accent-rose);
    }

    /* 공통 카드 스타일 */
    .card {
      background-color: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      padding: 20px 24px;
    }

    .card-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 14px;
      padding-bottom: 10px;
      border-bottom: 1px solid var(--border-color);
    }

    .card-title {
      font-size: 1.05rem;
      font-weight: 700;
      color: var(--accent-blue);
      display: flex;
      align-items: center;
      gap: 8px;
    }

    /* 알고리즘 복잡도 표시 그리드 */
    .complexity-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
    }

    .complexity-item {
      background: var(--bg-input);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 10px 14px;
    }

    .complexity-item .label {
      font-size: 0.76rem;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }

    .complexity-item .formula {
      font-family: var(--font-mono);
      font-size: 0.86rem;
      color: var(--accent-emerald);
      margin-top: 4px;
      font-weight: 600;
    }

    /* 폼 및 버튼 스타일 */
    .row {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
    }

    input[type="text"],
    input[type="number"],
    select,
    textarea {
      background: var(--bg-input);
      border: 1px solid var(--border-color);
      color: var(--text-main);
      padding: 10px 14px;
      border-radius: 8px;
      font-size: 0.92rem;
      font-family: var(--font-sans);
      outline: none;
      transition: border-color 0.15s;
    }

    input:focus,
    select:focus,
    textarea:focus {
      border-color: var(--accent-blue);
    }

    .btn {
      cursor: pointer;
      border: none;
      border-radius: 8px;
      padding: 10px 18px;
      font-size: 0.9rem;
      font-weight: 600;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      transition: opacity 0.15s, transform 0.1s;
    }

    .btn:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }

    .btn:active:not(:disabled) {
      transform: translateY(1px);
    }

    .btn-primary {
      background: var(--accent-indigo);
      color: #fff;
    }

    .btn-emerald {
      background: var(--accent-emerald);
      color: #052e16;
    }

    .btn-outline {
      background: transparent;
      border: 1px solid var(--border-color);
      color: var(--text-main);
    }

    /* 파일명 표시 배지 */
    .file-display-box {
      flex: 1;
      min-width: 260px;
      background: var(--bg-input);
      border: 1px dashed var(--border-color);
      border-radius: 8px;
      padding: 10px 14px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.9rem;
    }

    .file-name-tag {
      font-weight: 600;
      color: var(--accent-blue);
    }

    /* 진행 현황(Progress Bar & Stream Log) 영역 */
    .progress-wrapper {
      margin-top: 14px;
      background: var(--bg-input);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 14px;
    }

    .progress-bar-track {
      width: 100%;
      height: 10px;
      background: #1e293b;
      border-radius: 999px;
      overflow: hidden;
      margin: 8px 0 12px 0;
    }

    .progress-bar-fill {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, var(--accent-indigo), var(--accent-emerald));
      transition: width 0.3s ease;
    }

    .stream-log-list {
      list-style: none;
      max-height: 160px;
      overflow-y: auto;
      font-family: var(--font-mono);
      font-size: 0.8rem;
      color: var(--text-muted);
      display: flex;
      flex-direction: column;
      gap: 6px;
    }

    .stream-log-list li {
      padding: 4px 8px;
      border-left: 3px solid var(--accent-blue);
      background: rgba(56, 189, 248, 0.05);
    }

    /* 에러 표시 배너 */
    .error-banner {
      display: none;
      background: rgba(244, 63, 94, 0.12);
      border: 1px solid var(--accent-rose);
      border-radius: 10px;
      padding: 16px 20px;
      color: #fecdd3;
    }

    .error-banner.active {
      display: block;
    }

    .error-header {
      font-weight: 700;
      color: var(--accent-rose);
      margin-bottom: 6px;
      display: flex;
      justify-content: space-between;
    }

    .error-detail {
      font-family: var(--font-mono);
      font-size: 0.8rem;
      background: rgba(0, 0, 0, 0.35);
      padding: 8px 10px;
      border-radius: 6px;
      margin-top: 8px;
      white-space: pre-wrap;
      word-break: break-all;
    }

    /* 분리된 결과 표시 영역 (2열 레이아웃) */
    .results-split {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
    }

    @media (max-width: 960px) {
      .results-split {
        grid-template-columns: 1fr;
      }
    }

    /* 확장 쿼리 칩 */
    .multi-query-list {
      display: flex;
      flex-direction: column;
      gap: 8px;
      margin-bottom: 16px;
    }

    .mq-item {
      background: var(--bg-input);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 8px 12px;
      font-size: 0.85rem;
    }

    .mq-perspective {
      font-size: 0.75rem;
      font-weight: 700;
      color: var(--accent-amber);
      display: block;
    }

    /* RRF 검색 문서 카드 */
    .doc-card {
      background: var(--bg-input);
      border: 1px solid var(--border-color);
      border-radius: 10px;
      padding: 14px;
      margin-bottom: 12px;
    }

    .doc-card-header {
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 8px;
      padding-bottom: 8px;
      border-bottom: 1px dashed var(--border-color);
    }

    .badge {
      padding: 2px 8px;
      border-radius: 6px;
      font-size: 0.75rem;
      font-weight: 600;
      font-family: var(--font-mono);
    }

    .badge-rrf {
      background: rgba(16, 185, 129, 0.18);
      color: var(--accent-emerald);
      border: 1px solid var(--accent-emerald);
    }

    .badge-faiss {
      background: rgba(56, 189, 248, 0.15);
      color: var(--accent-blue);
    }

    .badge-bm25 {
      background: rgba(245, 158, 11, 0.15);
      color: var(--accent-amber);
    }

    .badge-source {
      background: rgba(99, 102, 241, 0.2);
      color: #c7d2fe;
    }

    .doc-snippet {
      font-size: 0.88rem;
      color: #e2e8f0;
      white-space: pre-wrap;
    }

    /* LLM 답변 및 출처 카드 */
    .llm-answer-box {
      background: var(--bg-input);
      border: 1px solid var(--border-color);
      border-radius: 10px;
      padding: 18px;
      font-size: 0.94rem;
      white-space: pre-wrap;
      line-height: 1.75;
    }

    .citation-pill {
      display: inline-block;
      background: rgba(99, 102, 241, 0.25);
      border: 1px solid var(--accent-indigo);
      color: #e0e7ff;
      padding: 1px 8px;
      border-radius: 6px;
      font-size: 0.8rem;
      font-weight: 600;
      margin: 2px;
    }

    .sources-footer {
      margin-top: 14px;
      padding-top: 12px;
      border-top: 1px solid var(--border-color);
    }
  </style>
</head>
<body>
  <div class="container">
    <!-- 1. 상단 서버 연동 상태 표시 헤더 -->
    <header class="header-bar">
      <div class="title-group">
        <h1>⚡ Advanced Hybrid RAG &amp; RRF Fusion System</h1>
        <p>FAISS Dense Vector + BM25 Sparse Keyword | 5-Way Multi-Query Expansion | Reciprocal Rank Fusion</p>
      </div>
      <div class="status-cluster">
        <div class="status-badge" title="백엔드 서버와의 실시간 연동 상태">
          <span id="serverStatusDot" class="status-dot"></span>
          <span id="serverStatusText">서버 상태 확인 중...</span>
        </div>
        <div class="status-badge" id="dbIndexBadge">인덱스: 미구축 (0 Chunks)</div>
        <div class="status-badge" id="latencyBadge">지연시간: - ms</div>
        <button class="btn btn-outline" onclick="refreshServerStatus()">🔄 연동 상태 확인</button>
      </div>
    </header>

    <!-- Cloud Run URL 환경변수 설정 바 -->
    <section class="card">
      <div class="row" style="justify-content: space-between;">
        <div class="row" style="flex: 1;">
          <strong style="font-size: 0.88rem; color: var(--text-muted);">Cloud Run 백엔드 URL (환경변수 CLOUD_RUN_URL):</strong>
          <input type="text" id="cloudRunUrlInput" style="flex: 1; min-width: 280px;" placeholder="http://localhost:8080 또는 https://your-service.a.run.app" value="http://localhost:8080" />
          <button class="btn btn-outline" onclick="updateCloudRunUrl()">환경변수 저장 및 연결</button>
        </div>
      </div>
    </section>

    <!-- 공통 에러 표시 영역 (에러 발생 시에만 활성화) -->
    <div id="errorBanner" class="error-banner" role="alert">
      <div class="error-header">
        <span id="errorTitle">⚠️ 오류가 발생했습니다</span>
        <button class="btn btn-outline" style="padding: 2px 10px; font-size: 0.78rem;" onclick="clearError()">닫기</button>
      </div>
      <div id="errorMessage" style="font-size: 0.9rem;"></div>
      <div id="errorDetail" class="error-detail"></div>
    </div>

    <!-- 2. 알고리즘 복잡도 표시 패널 -->
    <section class="card">
      <div class="card-header">
        <div class="card-title">📐 파이프라인 알고리즘 시간·공간 복잡도 (Algorithmic Complexity)</div>
        <span id="activeComplexityMode" style="font-size: 0.8rem; color: var(--text-muted);">실시간 메트릭 연동됨</span>
      </div>
      <div class="complexity-grid">
        <div class="complexity-item">
          <div class="label">1. 다중 쿼리 확장 (Multi-Query, M=5)</div>
          <div class="formula" id="compMultiQuery">Time: O(M) [M = 5 관점]</div>
        </div>
        <div class="complexity-item">
          <div class="label">2. FAISS Dense 벡터 검색</div>
          <div class="formula" id="compFaiss">HNSW: O(M · d · log N) / FlatL2: O(M · N · d)</div>
        </div>
        <div class="complexity-item">
          <div class="label">3. BM25 Sparse 키워드 검색</div>
          <div class="formula" id="compBm25">Time: O(M · |Q| · N)</div>
        </div>
        <div class="complexity-item">
          <div class="label">4. RRF 순위 융합 및 재정렬</div>
          <div class="formula" id="compRrf">Time: O(M · K + U log U), Space: O(N·d + U)</div>
        </div>
      </div>
    </section>

    <!-- 3. 파일명 표시, 파일 요청 버튼, 벡터 DB 생성 버튼 및 진행 현황 영역 -->
    <section class="card">
      <div class="card-header">
        <div class="card-title">📄 1단계: PDF 파일 요청 및 FAISS + BM25 하이브리드 벡터 DB 생성</div>
      </div>

      <div class="row" style="margin-bottom: 12px;">
        <!-- 파일명 표시 상자 -->
        <div class="file-display-box">
          <div>
            <span style="color: var(--text-muted); font-size: 0.82rem;">선택된 PDF 파일: </span>
            <span id="selectedFileName" class="file-name-tag">선택된 파일 없음</span>
          </div>
          <span id="selectedFileSize" style="font-size: 0.8rem; color: var(--text-muted);">0 KB</span>
        </div>

        <!-- Google Drive 검색어(선택) 및 파일 요청 버튼 -->
        <input type="text" id="driveSearchInput" placeholder="Drive PDF 파일명(비우면 최근 PDF)" style="width: 220px;" />
        <button class="btn btn-outline" id="btnRequestDriveFile" onclick="handleRequestPdfFile()">📂 파일 요청 (Drive/로컬)</button>
        <input type="file" id="localPdfInput" accept="application/pdf" style="display: none;" onchange="handleLocalPdfSelected(event)" />
        <button class="btn btn-outline" onclick="document.getElementById('localPdfInput').click()">💻 로컬 PDF 파일 선택</button>
      </div>

      <div class="row" style="justify-content: space-between;">
        <div class="row">
          <label style="font-size: 0.85rem; color: var(--text-muted);">FAISS 인덱스 방식:</label>
          <select id="faissIndexType">
            <option value="IndexHNSWFlat" selected>IndexHNSWFlat (계층 그래프 근사 탐색 - O(d log N))</option>
            <option value="IndexFlatL2">IndexFlatL2 (전수 L2 거리 탐색 - O(N d))</option>
          </select>

          <label style="font-size: 0.85rem; color: var(--text-muted);">청크 크기:</label>
          <input type="number" id="chunkSizeInput" value="500" min="150" max="2000" style="width: 90px;" />
        </div>

        <!-- 벡터 DB 생성 버튼 -->
        <button class="btn btn-emerald" id="btnBuildDb" onclick="handleBuildVectorDB()">
          🚀 벡터 DB 생성 (FAISS + BM25 인덱싱)
        </button>
      </div>

      <!-- 서버에서 RAG 생성 진행 현황 표시 영역 -->
      <div class="progress-wrapper">
        <div style="display: flex; justify-content: space-between; font-size: 0.85rem;">
          <span id="ragProgressStatus">대기 중 — PDF 파일을 요청하거나 선택한 뒤 [벡터 DB 생성] 버튼을 눌러주세요.</span>
          <span id="ragProgressPercent" style="font-weight: 700; color: var(--accent-emerald);">0%</span>
        </div>
        <div class="progress-bar-track">
          <div id="ragProgressBar" class="progress-bar-fill"></div>
        </div>
        <ul id="ragStreamLogs" class="stream-log-list">
          <li>[SYSTEM] StreamHandler 대기 중...</li>
        </ul>
      </div>
    </section>

    <!-- 4. 질의 입력 영역 및 질의하기 버튼 -->
    <section class="card">
      <div class="card-header">
        <div class="card-title">🔍 2단계: Multi-Query (5개 확장) + Hybrid Search + RRF 융합 질의하기</div>
      </div>

      <div class="row" style="margin-bottom: 12px;">
        <input
          type="text"
          id="userQueryInput"
          style="flex: 1; font-size: 1rem; padding: 12px 16px;"
          placeholder="질의 내용을 입력하세요 (예: 이 문서의 핵심 아키텍처와 주요 장단점은 무엇인가요?)"
          onkeydown="if(event.key === 'Enter') handleQueryRAG();"
        />
        <button class="btn btn-primary" id="btnQueryRag" onclick="handleQueryRAG()" style="padding: 12px 24px;">
          ✨ 질의하기 (Hybrid RAG Fusion)
        </button>
      </div>

      <div class="row" style="font-size: 0.84rem; color: var(--text-muted);">
        <span>상위 반환 문서 수 (Top-K):</span>
        <input type="number" id="topKInput" value="5" min="1" max="15" style="width: 70px; padding: 6px 10px;" />
        <span>RRF 상수 (k):</span>
        <input type="number" id="rrfKInput" value="60" min="1" max="120" style="width: 75px; padding: 6px 10px;" />
        <span>FAISS Dense 가중치:</span>
        <input type="number" id="denseWeightInput" value="1.0" step="0.1" min="0" max="5" style="width: 75px; padding: 6px 10px;" />
        <span>BM25 Sparse 가중치:</span>
        <input type="number" id="sparseWeightInput" value="1.0" step="0.1" min="0" max="5" style="width: 75px; padding: 6px 10px;" />
      </div>
    </section>

    <!-- 5. 답변 표시 영역 (RAG 검색 내용과 LLM 답변 분리 표시 + 출처 표기) -->
    <div class="results-split">
      <!-- 좌측: 다중 쿼리 확장(5가지) + RRF 재정렬 RAG 검색 내용 -->
      <section class="card">
        <div class="card-header">
          <div class="card-title">📚 RAG 검색 결과 (5-Way Multi-Query &amp; RRF 상위 스니펫)</div>
          <span id="searchTimingBadge" class="badge badge-faiss">대기 중</span>
        </div>

        <h4 style="font-size: 0.86rem; color: var(--text-muted); margin-bottom: 8px;">
          ① LLM 다중 쿼리 확장 계층 (5가지 관점 파생 쿼리)
        </h4>
        <div id="multiQueryContainer" class="multi-query-list">
          <div class="mq-item" style="color: var(--text-muted);">질의를 실행하면 5가지 관점으로 확장된 파생 쿼리가 여기에 표시됩니다.</div>
        </div>

        <h4 style="font-size: 0.86rem; color: var(--text-muted); margin: 14px 0 8px 0;">
          ② RRF (Reciprocal Rank Fusion) 재정렬 상위 문서 스니펫 및 메타데이터
        </h4>
        <div id="ragDocumentsContainer">
          <div class="doc-card" style="color: var(--text-muted);">
            FAISS Dense 유사도 순위와 BM25 키워드 매칭 순위를 RRF 공식으로 융합한 상위 문서 스니펫과 출처 메타데이터가 여기에 표시됩니다.
          </div>
        </div>
      </section>

      <!-- 우측: LLM 최종 답변 및 명확한 출처 표기 -->
      <section class="card">
        <div class="card-header">
          <div class="card-title">🤖 LLM 종합 답변 및 출처 (Citations)</div>
          <span id="llmTimingBadge" class="badge badge-rrf">대기 중</span>
        </div>

        <div id="llmAnswerContainer" class="llm-answer-box">
          질의를 실행하면 우측 영역에 RRF 상위 문서를 근거로 합성된 LLM의 최종 답변이 RAG 검색 스니펫과 분리되어 표시됩니다.
        </div>

        <div class="sources-footer">
          <div style="font-size: 0.84rem; font-weight: 700; color: var(--text-muted); margin-bottom: 8px;">
            📌 인용된 문서 출처 목록 (Verified Sources)
          </div>
          <div id="citationListContainer" style="display: flex; flex-wrap: wrap; gap: 6px;">
            <span style="font-size: 0.82rem; color: var(--text-muted);">아직 표시할 출처가 없습니다.</span>
          </div>
        </div>
      </section>
    </div>
  </div>

  <script>
    /**
     * ========================================================================
     * 클라이언트 상태 관리 및 GAS / 직접 HTTP 하이브리드 브릿지
     * ========================================================================
     * - Google Apps Script 웹앱 환경에서는 `google.script.run`을 통해 `Code.gs`의
     *   `StreamHandler` 및 `UrlFetchApp`을 호출합니다.
     * - 로컬 브라우저에서 직접 `index.html`을 열어 테스트할 때에도 즉시 동작하도록
     *   자동으로 `fetch()` 기반 클라이언트 `StreamHandler` 폴백을 지원합니다.
     */
    const AppState = {
      fileName: null,
      pdfBase64: null,
      fileSizeBytes: 0,
      isGasEnv: typeof google !== 'undefined' && google.script && google.script.run
    };

    /** 에러 배너 표시 함수 (정확한 단계, 에러 코드, 원인, 세부내용 표시) */
    function showError(stage, code, message, detail) {
      const banner = document.getElementById('errorBanner');
      document.getElementById('errorTitle').textContent = `⚠️ [${stage}] 오류 발생 (코드: ${code})`;
      document.getElementById('errorMessage').textContent = message || '알 수 없는 오류가 발생했습니다.';
      const detailEl = document.getElementById('errorDetail');
      if (detail) {
        detailEl.style.display = 'block';
        detailEl.textContent = detail;
      } else {
        detailEl.style.display = 'none';
      }
      banner.classList.add('active');
    }

    function clearError() {
      document.getElementById('errorBanner').classList.remove('active');
    }

    /** 진행률 바 및 스트림 로그 업데이트 함수 */
    function updateProgressUI(percent, statusText, logLine) {
      document.getElementById('ragProgressBar').style.width = `${Math.max(0, Math.min(100, percent))}%`;
      document.getElementById('ragProgressPercent').textContent = `${Math.round(percent)}%`;
      if (statusText) {
        document.getElementById('ragProgressStatus').textContent = statusText;
      }
      if (logLine) {
        const ul = document.getElementById('ragStreamLogs');
        const li = document.createElement('li');
        const now = new Date().toLocaleTimeString();
        li.textContent = `[${now}] ${logLine}`;
        ul.appendChild(li);
        ul.scrollTop = ul.scrollHeight;
      }
    }

    /** 알고리즘 복잡도 UI 패널 업데이트 */
    function updateComplexityUI(comp) {
      if (!comp) return;
      if (comp.multi_query_expansion || comp.multi_query) {
        document.getElementById('compMultiQuery').textContent = comp.multi_query_expansion || comp.multi_query;
      }
      if (comp.faiss_dense_search || comp.faiss_build || comp.faiss_search_per_query) {
        document.getElementById('compFaiss').textContent =
          comp.faiss_dense_search || comp.faiss_search_per_query || comp.faiss_build;
      }
      if (comp.bm25_sparse_search || comp.bm25_build) {
        document.getElementById('compBm25').textContent = comp.bm25_sparse_search || comp.bm25_build;
      }
      if (comp.rrf_fusion_rerank || comp.space_complexity) {
        document.getElementById('compRrf').textContent =
          (comp.rrf_fusion_rerank ? comp.rrf_fusion_rerank + ' | ' : '') + (comp.space_complexity || '');
      }
    }

    /**
     * 1. 서버 연동 상태 확인 (GAS checkServerHealth 또는 직접 /health 호출)
     */
    async function refreshServerStatus() {
      const dot = document.getElementById('serverStatusDot');
      const text = document.getElementById('serverStatusText');
      const dbBadge = document.getElementById('dbIndexBadge');
      const latencyBadge = document.getElementById('latencyBadge');

      text.textContent = '연결 확인 중...';

      if (AppState.isGasEnv) {
        google.script.run
          .withSuccessHandler((res) => applyHealthResponse(res))
          .withFailureHandler((err) => {
            dot.className = 'status-dot offline';
            text.textContent = 'GAS 오류';
            showError('GAS_BRIDGE', 'ERR_GAS_RPC', err.message || String(err));
          })
          .checkServerHealth();
      } else {
        const baseUrl = document.getElementById('cloudRunUrlInput').value.trim().replace(/\/+$/, '');
        const t0 = performance.now();
        try {
          const resp = await fetch(`${baseUrl}/health`);
          const latencyMs = Math.round(performance.now() - t0);
          if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
          const info = await resp.json();
          applyHealthResponse({
            status: 'online',
            cloudRunUrl: baseUrl,
            latencyMs,
            serverInfo: info
          });
        } catch (err) {
          dot.className = 'status-dot offline';
          text.textContent = '서버 오프라인 (미연결)';
          latencyBadge.textContent = '지연시간: - ms';
        }
      }
    }

    function applyHealthResponse(res) {
      const dot = document.getElementById('serverStatusDot');
      const text = document.getElementById('serverStatusText');
      const dbBadge = document.getElementById('dbIndexBadge');
      const latencyBadge = document.getElementById('latencyBadge');

      if (res.cloudRunUrl) {
        document.getElementById('cloudRunUrlInput').value = res.cloudRunUrl;
      }

      if (res.status === 'online' && res.serverInfo) {
        dot.className = 'status-dot online';
        text.textContent = `서버 연결됨 (${res.serverInfo.engine_mode})`;
        latencyBadge.textContent = `지연시간: ${res.latencyMs} ms`;
        if (res.serverInfo.db_ready) {
          dbBadge.textContent = `인덱스: ${res.serverInfo.index_type} (${res.serverInfo.total_chunks} Chunks | ${res.serverInfo.active_file_name})`;
        } else {
          dbBadge.textContent = '인덱스: 미구축 (0 Chunks)';
        }
      } else {
        dot.className = 'status-dot offline';
        text.textContent = '서버 오프라인';
        if (res.error) {
          showError(res.error.stage, res.error.error_code, res.error.message, res.error.detail);
        }
      }
    }

    /** Cloud Run URL 환경변수 업데이트 */
    function updateCloudRunUrl() {
      clearError();
      const newUrl = document.getElementById('cloudRunUrlInput').value.trim();
      if (AppState.isGasEnv) {
        google.script.run
          .withSuccessHandler((res) => applyHealthResponse(res))
          .withFailureHandler((err) => showError('ENV_SAVE', 'ERR_SAVE_URL', err.message))
          .saveCloudRunUrl(newUrl);
      } else {
        refreshServerStatus();
      }
    }

    /**
     * 2. 파일 요청 버튼 처리 (GAS 환경에서는 Google Drive PDF 요청, 로컬에서는 파일 선택창 오픈)
     */
    function handleRequestPdfFile() {
      clearError();
      const keyword = document.getElementById('driveSearchInput').value.trim();

      if (AppState.isGasEnv) {
        updateProgressUI(5, 'Google Drive에서 PDF 파일을 요청하는 중...', `[DRIVE] 파일 검색어: "${keyword || '최근 PDF'}"`);
        google.script.run
          .withSuccessHandler((res) => {
            if (res.status === 'error') {
              showError(res.error.stage, res.error.error_code, res.error.message, res.error.detail);
              updateProgressUI(0, '파일 요청 실패', `[ERROR] ${res.error.message}`);
              return;
            }
            AppState.fileName = res.fileName;
            AppState.pdfBase64 = res.pdfBase64;
            AppState.fileSizeBytes = res.fileSizeBytes;
            document.getElementById('selectedFileName').textContent = res.fileName;
            document.getElementById('selectedFileSize').textContent = `${(res.fileSizeBytes / 1024).toFixed(1)} KB`;
            updateProgressUI(15, `'${res.fileName}' 파일 준비 완료. [벡터 DB 생성]을 눌러주세요.`, `[DRIVE] '${res.fileName}' 로드 성공`);
          })
          .withFailureHandler((err) => {
            showError('DRIVE_RPC', 'ERR_GAS_DRIVE_CALL', err.message || String(err));
          })
          .requestPdfFromDrive(keyword);
      } else {
        // 로컬 브라우저에서 테스트 시 로컬 PDF 선택 다이얼로그 호출
        document.getElementById('localPdfInput').click();
      }
    }

    /** 로컬 PDF 파일 선택 시 Base64 인코딩 및 파일명 표시 */
    function handleLocalPdfSelected(event) {
      clearError();
      const file = event.target.files && event.target.files[0];
      if (!file) return;

      const reader = new FileReader();
      reader.onload = function(e) {
        const dataUrl = e.target.result;
        const base64Content = dataUrl.split(',')[1];
        AppState.fileName = file.name;
        AppState.pdfBase64 = base64Content;
        AppState.fileSizeBytes = file.size;

        document.getElementById('selectedFileName').textContent = file.name;
        document.getElementById('selectedFileSize').textContent = `${(file.size / 1024).toFixed(1)} KB`;
        updateProgressUI(15, `'${file.name}' 로드 완료. [벡터 DB 생성] 버튼을 클릭하세요.`, `[FILE] 로컬 PDF '${file.name}' (${(file.size / 1024).toFixed(1)} KB) 준비됨`);
      };
      reader.onerror = function() {
        showError('LOCAL_FILE_READ', 'ERR_FILE_READER', '선택한 PDF 파일을 읽는 중 오류가 발생했습니다.');
      };
      reader.readAsDataURL(file);
    }

    /**
     * 3. 벡터 DB 생성 요청 (StreamHandler 연동)
     */
    async function handleBuildVectorDB() {
      clearError();
      if (!AppState.pdfBase64) {
        showError(
          'BUILD_DB_PRECHECK',
          'ERR_NO_PDF_LOADED',
          '먼저 [파일 요청] 또는 [로컬 PDF 파일 선택] 버튼으로 PDF 파일을 불러와 주세요.'
        );
        return;
      }

      const indexType = document.getElementById('faissIndexType').value;
      const chunkSize = parseInt(document.getElementById('chunkSizeInput').value, 10) || 500;
      const btn = document.getElementById('btnBuildDb');
      btn.disabled = true;

      document.getElementById('ragStreamLogs').innerHTML = '';
      updateProgressUI(10, '서버로 PDF 전송 및 하이브리드 인덱스 생성 요청 중...', `[START] '${AppState.fileName}' -> FAISS(${indexType}) + BM25 인덱싱 시작`);

      if (AppState.isGasEnv) {
        google.script.run
          .withSuccessHandler((res) => {
            btn.disabled = false;
            processStreamEventsResponse(res, 'BUILD_DB');
          })
          .withFailureHandler((err) => {
            btn.disabled = false;
            showError('BUILD_DB_RPC', 'ERR_GAS_BUILD_DB', err.message || String(err));
          })
          .buildVectorDBOnServer({
            fileName: AppState.fileName,
            pdfBase64: AppState.pdfBase64,
            indexType: indexType,
            chunkSize: chunkSize,
            chunkOverlap: 100
          });
      } else {
        // 브라우저 직접 NDJSON 실시간 스트리밍 수신 (ReadableStream)
        const baseUrl = document.getElementById('cloudRunUrlInput').value.trim().replace(/\/+$/, '');
        try {
          const response = await fetch(`${baseUrl}/build-db`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              file_name: AppState.fileName,
              pdf_base64: AppState.pdfBase64,
              index_type: indexType,
              chunk_size: chunkSize,
              chunk_overlap: 100,
              stream: true
            })
          });

          await consumeBrowserNdjsonStream(response, 'BUILD_DB');
          refreshServerStatus();
        } catch (err) {
          showError('NETWORK_BUILD_DB', 'ERR_FETCH_BUILD_DB', '벡터 DB 생성 중 서버 통신 오류가 발생했습니다.', String(err));
        } finally {
          btn.disabled = false;
        }
      }
    }

    /**
     * 4. 하이브리드 Multi-Query + RRF 질의 요청
     */
    async function handleQueryRAG() {
      clearError();
      const query = document.getElementById('userQueryInput').value.trim();
      if (!query) {
        showError('QUERY_PRECHECK', 'ERR_EMPTY_QUERY', '질의 입력창에 질문 내용을 입력해 주세요.');
        return;
      }

      const topK = parseInt(document.getElementById('topKInput').value, 10) || 5;
      const rrfK = parseInt(document.getElementById('rrfKInput').value, 10) || 60;
      const denseWeight = parseFloat(document.getElementById('denseWeightInput').value) || 1.0;
      const sparseWeight = parseFloat(document.getElementById('sparseWeightInput').value) || 1.0;

      const btn = document.getElementById('btnQueryRag');
      btn.disabled = true;
      updateProgressUI(15, '5가지 다중 쿼리 확장 및 FAISS + BM25 하이브리드 검색 진행 중...', `[QUERY] "${query}"`);

      if (AppState.isGasEnv) {
        google.script.run
          .withSuccessHandler((res) => {
            btn.disabled = false;
            processStreamEventsResponse(res, 'QUERY');
          })
          .withFailureHandler((err) => {
            btn.disabled = false;
            showError('QUERY_RPC', 'ERR_GAS_QUERY', err.message || String(err));
          })
          .queryHybridRAGOnServer({
            query,
            topK,
            rrfK,
            denseWeight,
            sparseWeight
          });
      } else {
        const baseUrl = document.getElementById('cloudRunUrlInput').value.trim().replace(/\/+$/, '');
        try {
          const response = await fetch(`${baseUrl}/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              query,
              top_k: topK,
              rrf_k: rrfK,
              dense_weight: denseWeight,
              sparse_weight: sparseWeight,
              stream: true
            })
          });
          await consumeBrowserNdjsonStream(response, 'QUERY');
        } catch (err) {
          showError('NETWORK_QUERY', 'ERR_FETCH_QUERY', '질의 요청 중 서버 통신 오류가 발생했습니다.', String(err));
        } finally {
          btn.disabled = false;
        }
      }
    }

    /** GAS StreamHandler 응답 처리 */
    function processStreamEventsResponse(res, mode) {
      if (!res) return;
      if (res.events && Array.isArray(res.events)) {
        res.events.forEach((evt) => handleSingleStreamEvent(evt, mode));
      }
      if (res.status === 'error' && res.error) {
        showError(res.error.stage || mode, res.error.error_code || 'ERR_SERVER', res.error.message, res.error.detail);
        return;
      }
      if (res.complexity) {
        updateComplexityUI(res.complexity);
      }
      if (mode === 'BUILD_DB') {
        refreshServerStatus();
      } else if (mode === 'QUERY' && res.data) {
        renderQueryResult(res.data);
      }
    }

    /** 로컬 브라우저 ReadableStream 기반 실시간 NDJSON 스트림 파서 */
    async function consumeBrowserNdjsonStream(response, mode) {
      if (!response.ok) {
        const errText = await response.text();
        showError(`HTTP_${response.status}`, 'ERR_HTTP_STATUS', `서버가 HTTP ${response.status} 응답을 반환했습니다.`, errText);
        return;
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        for (const line of lines) {
          if (!line.trim()) continue;
          const evt = JSON.parse(line);
          handleSingleStreamEvent(evt, mode);
        }
      }
      if (buffer.trim()) {
        const evt = JSON.parse(buffer.trim());
        handleSingleStreamEvent(evt, mode);
      }
    }

    /** 단일 StreamHandler 이벤트 프레임 처리 */
    function handleSingleStreamEvent(evt, mode) {
      if (evt.complexity) {
        updateComplexityUI(evt.complexity);
      }
      if (evt.event === 'error') {
        const err = evt.error || {};
        updateProgressUI(0, `오류 중단: ${err.message || evt.message}`, `[ERROR] (${err.error_code}) ${err.message}`);
        showError(err.stage || evt.stage || mode, err.error_code || 'ERR_STREAM', err.message || evt.message, err.detail);
        return;
      }

      updateProgressUI(evt.progress || 0, evt.message, `[${evt.stage}] ${evt.message}`);

      if (mode === 'QUERY') {
        if (evt.event === 'partial_result' && evt.data) {
          if (evt.data.expanded_queries) renderMultiQueries(evt.data.expanded_queries);
          if (evt.data.ranked_documents) renderRankedDocuments(evt.data.ranked_documents);
        } else if (evt.event === 'complete' && evt.data) {
          renderQueryResult(evt.data);
        }
      }
    }

    /** 5가지 확장된 다중 쿼리 렌더링 */
    function renderMultiQueries(queries) {
      const container = document.getElementById('multiQueryContainer');
      if (!queries || !queries.length) return;
      container.innerHTML = queries
        .map(
          (q) => `
          <div class="mq-item">
            <span class="mq-perspective">${escapeHtml(q.perspective)}</span>
            <div>${escapeHtml(q.query)}</div>
          </div>`
        )
        .join('');
    }

    /** RRF 순위 융합 상위 문서 스니펫 및 메타데이터 렌더링 */
    function renderRankedDocuments(docs) {
      const container = document.getElementById('ragDocumentsContainer');
      if (!docs || !docs.length) {
        container.innerHTML = '<div class="doc-card">검색된 문서가 없습니다.</div>';
        return;
      }
      container.innerHTML = docs
        .map(
          (doc) => `
          <div class="doc-card">
            <div class="doc-card-header">
              <div>
                <span class="badge badge-rrf">RRF #${doc.final_rank} (점수: ${Number(doc.rrf_score).toFixed(5)})</span>
                <span class="badge badge-faiss">FAISS 순위: ${doc.best_dense_rank ?? '-'} (유사도: ${doc.dense_similarity})</span>
                <span class="badge badge-bm25">BM25 순위: ${doc.best_sparse_rank ?? '-'} (점수: ${doc.bm25_score})</span>
              </div>
              <span class="badge badge-source">${escapeHtml(doc.citation)}</span>
            </div>
            <div class="doc-snippet">${escapeHtml(doc.content)}</div>
          </div>`
        )
        .join('');
    }

    /** 전체 질의 결과(RAG 스니펫 + LLM 답변 + 출처 목록 + 타이밍 메트릭) 렌더링 */
    function renderQueryResult(result) {
      if (result.expanded_queries) renderMultiQueries(result.expanded_queries);
      if (result.ranked_documents) renderRankedDocuments(result.ranked_documents);

      // 타이밍 및 복잡도 배지 업데이트
      if (result.metrics) {
        if (result.metrics.complexity) updateComplexityUI(result.metrics.complexity);
        const t = result.metrics.timings_ms || {};
        document.getElementById('searchTimingBadge').textContent =
          `Multi-Query: ${t.multi_query_ms || 0}ms | Hybrid+RRF: ${(t.hybrid_search_ms || 0) + (t.rrf_fusion_ms || 0)}ms`;
        document.getElementById('llmTimingBadge').textContent =
          `LLM 합성: ${t.llm_synthesis_ms || 0}ms (총 ${t.total_ms || 0}ms)`;
      }

      // LLM 답변 렌더링 ([출처: ...] 패턴을 시각적 배지로 강조)
      const rawAnswer = result.llm_answer || '';
      const safeAnswer = escapeHtml(rawAnswer).replace(
        /\[출처:[^\]]+\]/g,
        (match) => `<span class="citation-pill">${match}</span>`
      );
      document.getElementById('llmAnswerContainer').innerHTML = safeAnswer;

      // 하단 출처 목록 렌더링
      const sourcesContainer = document.getElementById('citationListContainer');
      const sources = result.sources || [];
      if (sources.length > 0) {
        sourcesContainer.innerHTML = sources
          .map(
            (s) =>
              `<span class="citation-pill" title="RRF Score: ${s.rrf_score}">${escapeHtml(s.citation)} (RRF: ${Number(s.rrf_score).toFixed(4)})</span>`
          )
          .join('');
      }
    }

    function escapeHtml(str) {
      if (str === null || str === undefined) return '';
      return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
    }

    // 페이지 로드 시 서버 연동 상태 자동 점검
    window.addEventListener('DOMContentLoaded', () => {
      refreshServerStatus();
    });
  </script>
</body>
</html>

