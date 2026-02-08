import uuid
import os
from typing import List, Dict, Any
from pathlib import Path
from unstructured.partition.auto import partition
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

class DocumentProcessor:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.persist_directory = Path("chroma_db")
        self.persist_directory.mkdir(exist_ok=True)
        self.collection_prefix = "logistics_docs"
    
    def process_document(self, file_path: str, filename: str) -> Dict[str, Any]:
        doc_id = str(uuid.uuid4())
        
        # 1. Improved Partitioning: Extracting as elements to keep structure
        elements = partition(filename=file_path)
        
        # 2. Group elements into logical blocks (Headers + Tables + Text)
        # Logistics docs have many short lines; joining them with double newlines
        # helps the embedding model see them as 'fields'.
        full_text = ""
        for el in elements:
            element_text = str(el).strip()
            if element_text:
                full_text += f"{element_text}\n"

        if not full_text.strip():
            return {"document_id": doc_id, "status": "error", "error": "Empty document"}

        # 3. Enhanced Chunking for Logistics
        # We decrease chunk size but use smarter separators to keep key-value pairs together
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, # Smaller chunks are better for specific field retrieval
            chunk_overlap=50,
            separators=["\n\n", "\n", "|", ":", ". ", " ", ""],
            keep_separator=True
        )
        chunks = text_splitter.split_text(full_text)
        
        documents = [
            Document(
                page_content=chunk,
                metadata={
                    "document_id": doc_id,
                    "filename": filename,
                    "chunk_index": i,
                    "source": filename
                }
            )
            for i, chunk in enumerate(chunks)
        ]
        
        collection_name = f"{self.collection_prefix}_{doc_id}"
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_directory)
        )
        vectorstore.add_documents(documents)
        
        return {
            "document_id": doc_id,
            "filename": filename,
            "status": "processed",
            "chunk_count": len(chunks)
        }
