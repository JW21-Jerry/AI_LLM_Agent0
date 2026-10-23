import os
import io
import json
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict

# LangChain 및 AI 모델 관련 임포트
from langchain_community.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from supabase.client import create_client, Client

# 환경 변수에서 키 로드 (Cloud Run 환경변수에 등록 필요)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY") # RLS를 우회하기 위해 Service Role Key 사용 권장

# Supabase 클라이언트 초기화
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# 임베딩 모델 초기화 (OpenAI)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# LLM 모델 초기화 (Gemini) - 스트리밍 지원 활성화
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=GOOGLE_API_KEY,
    streaming=True, # 스트리밍 핸들러 적용을 위한 설정
    temperature=0
)

app = FastAPI()

# GAS(웹브라우저 클라이언트)에서 직접 호출할 수 있도록 CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 보안을 위해 실제 서비스 시 GAS url로 제한하는 것이 좋습니다.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str
    chat_history: List[Dict[str, str]] # [{'role': 'user', 'content': '...'}, {'role': 'ai', 'content': '...'}]

@app.get("/")
async def root():
    return {"status": "Server is running"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """PDF 파일을 업로드받아 텍스트 추출, 청크 분할, 임베딩 후 Supabase에 저장합니다."""
    try:
        # 1. PDF 텍스트 추출
        pdf_reader = PdfReader(file.file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
            
        # 2. 청크 단위로 쪼개기
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(text)
        
        # 메타데이터 생성
        metadatas = [{"source": file.filename, "chunk_index": i} for i in range(len(chunks))]
        
        # 3. Vector DB (Supabase) 에 저장 (내부적으로 OpenAI 임베딩 수행)
        vector_store = SupabaseVectorStore.from_texts(
            chunks,
            embeddings,
            client=supabase,
            table_name="documents",
            metadatas=metadatas
        )
        
        return {"status": "success", "message": f"{file.filename} 처리 완료. {len(chunks)}개의 청크가 벡터DB에 저장되었습니다.", "extracted_text": text[:500] + "...(생략)"}
    
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """질의 확장 후 Supabase에서 검색하고, Gemini 모델을 통해 답변을 스트리밍합니다."""
    
    # 1. 대화 기록(Chat History) 객체 변환
    formatted_history = []
    for msg in request.chat_history:
        if msg["role"] == "user":
            formatted_history.append(HumanMessage(content=msg["content"]))
        else:
            formatted_history.append(AIMessage(content=msg["content"]))

    # 2. Query Augmentation (질의 확장) 설정
    # 시스템 메시지: 대화 기록을 기반으로 질문 의도 파악 및 명확화
    augmentation_prompt = ChatPromptTemplate.from_messages([
        ("system", "기존의 대화 내용을 활용하여 사용자가 질문한 의도를 파악해서 한문장의 명료한 질문으로 변환하라. 대명사나 '이', '저', '그'와 같은 표현을 명확한 명사로 표현하라."),
        MessagesPlaceholder(variable_name="chat_history"), # 대화 기록 전달 공간 마련
        ("human", "{question}")
    ])
    
    # 질의 확장 체인 구성
    query_augmentation_chain = augmentation_prompt | llm | StrOutputParser()
    
    # 3. 체인 실행 (데이터 주입)하여 증강된 질의 획득
    if len(formatted_history) > 0:
        augmented_query = query_augmentation_chain.invoke({
            "chat_history": formatted_history,
            "question": request.query
        })
    else:
        augmented_query = request.query # 히스토리가 없으면 원본 질의 사용

    # 4. 증강된 질의로 벡터 DB 검색 (Retriever)
    vector_store = SupabaseVectorStore(
        embedding=embeddings,
        client=supabase,
        table_name="documents",
        query_name="match_documents"
    )
    docs = vector_store.similarity_search(augmented_query, k=3)
    
    context_text = "\n\n".join([f"[출처: {d.metadata.get('source', 'Unknown')}]\n{d.page_content}" for d in docs])

    # 5. 최종 RAG 프롬프트 설계
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", "너는 문서에 기반해 답변하는 도시 정책 전문가야. 제공된 Context만을 기반으로 상세히 답변하고, Context에 없는 내용은 모른다고 답해라.\n\nContext:\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    rag_chain = rag_prompt | llm | StrOutputParser()

    # 6. 실시간 스트리밍 핸들러 적용 (Server-Sent Events 방식)
    async def generate_stream():
        # 먼저 클라이언트에게 증강된 질문과 검색된 출처(Context)를 메타데이터로 보냅니다.
        meta_info = {
            "type": "meta",
            "augmented_query": augmented_query,
            "context": context_text
        }
        yield f"data: {json.dumps(meta_info)}\n\n"

        # LLM 답변 스트리밍 생성
        async for chunk in rag_chain.astream({
            "context": context_text,
            "chat_history": formatted_history,
            "question": request.query
        }):
            # 청크 데이터를 JSON 형태로 스트리밍
            chunk_data = {"type": "chunk", "content": chunk}
            yield f"data: {json.dumps(chunk_data)}\n\n"
            
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate_stream(), media_type="text/event-stream")
