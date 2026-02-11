from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import uuid
import os
from ..models.schemas import *
from ..core.document_processor import DocumentProcessor
from ..core.rag_pipeline import RAGPipeline
from ..core.structured_extraction import StructuredExtractor

app = FastAPI(title="Ultra Doc-Intelligence API")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

processor = DocumentProcessor()
rag = RAGPipeline()
extractor = StructuredExtractor()

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(('.pdf', '.docx', '.txt')):
        raise HTTPException(400, "Only PDF, DOCX, TXT allowed")
    id_info = uuid.uuid4()
    file_path = os.path.join(UPLOAD_DIR, f"{id_info}_{file.filename}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    actual_filename = f"{id_info}_{file.filename}"
    result = processor.process_document(file_path, actual_filename)
    return result

@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    if not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(500, "GEMINI_API_KEY not configured")
    result = rag.ask(request.question, request.document_id)
    print(result)
    return AskResponse(**result)

@app.post("/extract", response_model=Dict)
async def extract_structured(request: ExtractRequest):
    contexts = rag.retrieve("", request.document_id, top_k=20)
    return extractor.extract(contexts)
