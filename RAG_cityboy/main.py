import os
import json
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import create_client, Client

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# FastAPI 앱 생성 및 CORS 설정
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 환경 변수 로드 (Cloud Run 환경 변수에 등록 필요)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # 구글 AI API 키
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # 오픈 AI API 키 (임베딩용)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Supabase 클라이언트 및 임베딩 모델(OpenAI) 초기화
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# LLM 객체 초기화 (gemini-3.5-flash-lite 적용)
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", google_api_key=GOOGLE_API_KEY, temperature=0.3)

# 클라이언트 요청 데이터 모델
class ChatRequest(BaseModel):
    query: str
    chat_history: List[Dict[str, str]] # [{'role': 'user', 'content': '...'}, ...]

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """PDF 파일을 업로드받아 텍스트 추출, 청크 분할 후 Supabase Vector DB에 적재합니다."""
    file_path = f"/tmp/{file.filename}"
    
    # 1. 파일 임시 저장
    with open(file_path, "wb") as f:
        f.write(await file.read())
        
    # 2. PDF 읽기 및 텍스트 변환
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    # 클라이언트 화면에 표시할 파일 전체 내용 추출
    full_text = "\n".join([doc.page_content for doc in documents])
    
    # 3. 텍스트를 청크 단위로 쪼개기
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    # 4. 벡터 DB 생성: Supabase 테이블("documents")에 저장
    SupabaseVectorStore.from_documents(
        splits,
        embeddings,
        client=supabase,
        table_name="documents",
        query_name="match_documents"
    )
    
    os.remove(file_path) # 임시 파일 삭제
    return {"status": "success", "file_name": file.filename, "content": full_text, "chunks": len(splits)}

@app.post("/chat")
async def chat_stream(request: ChatRequest):
    """대화 이력을 통한 질의 확장 및 Supabase 검색 후 실시간 답변을 생성합니다."""
    
    # 1. 과거 대화 내역 포맷팅
    history = []
    for msg in request.chat_history:
        if msg['role'] == 'user':
            history.append(HumanMessage(content=msg['content']))
        else:
            history.append(AIMessage(content=msg['content']))

    # 2. 시스템 메시지 및 프롬프트 설계 (의도 파악 및 질문 변환)
    augmentation_prompt = ChatPromptTemplate.from_messages([
        ("system", "기존의 대화 내용을 활용하여 사용자가 질문한 의도를 파악해서 한문장의 명료한 질문으로 변환하라. 대명사나 이, 저, 그와 같은 표현을 명확한 명사로 표현하라."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    
    # 3. Query Augmentation Chain 구성
    query_augmentation_chain = augmentation_prompt | llm | StrOutputParser()
    
    # 4. 체인 실행 (데이터 주입): 과거 기록과 현재 질문 주입
    augmented_query = query_augmentation_chain.invoke({
        "chat_history": history,
        "question": request.query
    })

    # 5. 증강된 질의로 벡터 DB 검색 (Supabase Retriever)
    vectorstore = SupabaseVectorStore(
        client=supabase,
        embedding=embeddings,
        table_name="documents",
        query_name="match_documents"
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(augmented_query)
    
    context_text = "\n\n".join([doc.page_content for doc in docs])
    sources_info = [f"출처 페이지: {doc.metadata.get('page', '알수없음')}" for doc in docs]

    # 6. RAG 답변 생성을 위한 시스템 메시지 설계
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", "너는 문서에 기반해 답변하는 도시 정책 전문가야.\n\n참고 문서:\n{context}"),
        ("human", "{question}")
    ])
    
    rag_chain = rag_prompt | llm | StrOutputParser()

    async def generate_response():
        # 검색 내역, 출처, 확장된 질문을 먼저 클라이언트로 전송
        meta_data = {
            "type": "meta",
            "augmented_query": augmented_query,
            "sources": sources_info,
            "context": context_text
        }
        yield f"data: {json.dumps(meta_data)}\n\n"
        
        # 실시간 스트리밍 핸들러 (StreamHandler) 적용
        async for chunk in rag_chain.astream({"context": context_text, "question": augmented_query}):
            yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
            
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate_response(), media_type="text/event-stream")
