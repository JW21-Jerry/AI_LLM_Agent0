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
from langchain.retrievers.multi_query import MultiQueryRetriever
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

    retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(), llm=llm
    )

    template = """당신은 도시 계획 설계자입니다. 특히 서울시에 특화된 설계자로 사용자의 질문에 답변하세요.
    아래의 제공된 문서 내용을 바탕으로 답변을 작성하세요.

    참고 문서:
    {context}

    사용자 질문: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    async def generate_response():
        yield f"data: {json.dumps({'type': 'meta', 'status': 'started'})}\n\n"
        async for chunk in rag_chain.astream(request.query):
            yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate_response(), media_type="text/event-stream")
