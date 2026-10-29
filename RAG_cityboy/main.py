import os
import json
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict

# 에러 우회를 위한 구글 직통 라이브러리 추가
import google.generativeai as genai
from langchain_core.embeddings import Embeddings

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import create_client, Client

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False, 
    allow_methods=["*"],
    allow_headers=["*"],
)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") 
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------------------------------------------------
# [수정됨] 404 에러를 우회하는 커스텀 구글 임베딩 클래스 (768차원)
genai.configure(api_key=GOOGLE_API_KEY)

class DirectGoogleEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [genai.embed_content(model="models/text-embedding-004", content=t)["embedding"] for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return genai.embed_content(model="models/text-embedding-004", content=text)["embedding"]

embeddings = DirectGoogleEmbeddings()
# ---------------------------------------------------------

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", google_api_key=GOOGLE_API_KEY, temperature=0.3)

class ChatRequest(BaseModel):
    query: str
    chat_history: List[Dict[str, str]]

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = f"/tmp/{file.filename}"
    
    with open(file_path, "wb") as f:
        f.write(await file.read())
        
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    full_text = "\n".join([doc.page_content for doc in documents])
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    SupabaseVectorStore.from_documents(
        splits,
        embeddings,
        client=supabase,
        table_name="documents",
        query_name="match_documents"
    )
    
    os.remove(file_path)
    return {"status": "success", "file_name": file.filename, "content": full_text, "chunks": len(splits)}

@app.post("/chat")
async def chat_stream(request: ChatRequest):
    history = []
    for msg in request.chat_history:
        if msg['role'] == 'user':
            history.append(HumanMessage(content=msg['content']))
        else:
            history.append(AIMessage(content=msg['content']))

    augmentation_prompt = ChatPromptTemplate.from_messages([
        ("system", "기존의 대화 내용을 활용하여 사용자가 질문한 의도를 파악해서 한문장의 명료한 질문으로 변환하라. 대명사나 이, 저, 그와 같은 표현을 명확한 명사로 표현하라."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    
    query_augmentation_chain = augmentation_prompt | llm | StrOutputParser()
    
    augmented_query = query_augmentation_chain.invoke({
        "chat_history": history,
        "question": request.query
    })

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

    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", "너는 문서에 기반해 답변하는 도시 정책 전문가야.\n\n참고 문서:\n{context}"),
        ("human", "{question}")
    ])
    
    rag_chain = rag_prompt | llm | StrOutputParser()

    async def generate_response():
        meta_data = {
            "type": "meta",
            "augmented_query": augmented_query,
            "sources": sources_info,
            "context": context_text
        }
        yield f"data: {json.dumps(meta_data)}\n\n"
        
        async for chunk in rag_chain.astream({"context": context_text, "question": augmented_query}):
            yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
            
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate_response(), media_type="text/event-stream")
