import os
import json
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

import google.generativeai as genai
from langchain_core.embeddings import Embeddings

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma

# 에러가 나는 모듈 대신 코어를 직접 사용
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False, 
    allow_methods=["*"],
    allow_headers=["*"],
)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") 

genai.configure(api_key=GOOGLE_API_KEY)
class DirectGoogleEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [genai.embed_content(model="models/text-embedding-004", content=t)["embedding"] for t in texts]
    def embed_query(self, text: str) -> List[float]:
        return genai.embed_content(model="models/text-embedding-004", content=text)["embedding"]

embeddings = DirectGoogleEmbeddings()
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", google_api_key=GOOGLE_API_KEY, temperature=0.3)

class ChatRequest(BaseModel):
    query: str

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = f"/tmp/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())
        
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory="/tmp/chroma_db"
    )
    os.remove(file_path)
    return {"status": "success", "file_name": file.filename, "chunks": len(splits)}

@app.post("/chat")
async def chat_stream(request: ChatRequest):
    vectorstore = Chroma(persist_directory="/tmp/chroma_db", embedding_function=embeddings)
    
    # ----------------------------------------------------
    # [에러 우회] 다중 질문(MultiQuery) 로직 직접 구현
    # 사용자의 질문을 3개로 다각화하여 검색 정확도를 높임
    multi_query_prompt = ChatPromptTemplate.from_messages([
        ("system", "사용자의 질문을 다각도로 분석하여 3개의 다른 형태의 질문으로 만들어주세요. 줄바꿈으로 구분하세요."),
        ("human", "{question}")
    ])
    multi_queries_str = (multi_query_prompt | llm | StrOutputParser()).invoke({"question": request.query})
    queries = multi_queries_str.split("\n") + [request.query]
    
    # 생성된 다중 질문으로 문서를 각각 검색 후 병합
    retrieved_docs = []
    retriever = vectorstore.as_retriever()
    for q in queries:
        if q.strip():
            retrieved_docs.extend(retriever.invoke(q))
    
    # 중복 문서 제거
    unique_docs = {doc.page_content: doc for doc in retrieved_docs}.values()
    context_text = "\n\n".join([doc.page_content for doc in unique_docs])
    # ----------------------------------------------------

    template = """당신은 도시 계획 설계자입니다. 특히 서울시에 특화된 설계자로 사용자의 질문에 답변하세요.
    아래의 제공된 문서 내용을 바탕으로 답변을 작성하세요.
    
    참고 문서:
    {context}
    
    사용자 질문: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    rag_chain = (
        prompt | llm | StrOutputParser()
    )

    async def generate_response():
        yield f"data: {json.dumps({'type': 'meta', 'status': 'started'})}\n\n"
        
        # RunnablePassthrough 처럼 context와 question을 주입하여 스트리밍
        async for chunk in rag_chain.astream({"context": context_text, "question": request.query}):
            yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
            
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate_response(), media_type="text/event-stream")
