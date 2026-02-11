import os
import numpy as np
from typing import List, Dict, Any
from pathlib import Path
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv

load_dotenv()

class RAGPipeline:
    def __init__(self):
        self.llm = ChatOllama(model="llama3.2:3b", temperature=0.0, num_predict=512)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.persist_directory = Path("chroma_db")
        self.collection_prefix = "logistics_docs"
        
        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", """"You are a specialized logistics document assistant. "
                "Your task is to answer user queries using ONLY the provided context snippets. "
                "If the specific information (like weight, price, or location) is not in the context, "
                "respond exactly with: 'Not found in document'. "
                "Do not use external knowledge.\n\n"
                f"### CONTEXT START ###\n{context}\n### CONTEXT END ###"""),
            ("human", "{question}")
        ])
        self.chain = self.rag_prompt | self.llm | StrOutputParser()
    
    def retrieve(self, question: str, document_id: str, top_k: int = 6) -> List[Dict]:
        """✅ REAL similarity scores (0.1-1.0)"""
        try:
            collection_name = f"{self.collection_prefix}_{document_id}"
            vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(self.persist_directory)
            )
            
            # Get ACTUAL similarity scores
            docs_with_scores = vectorstore.similarity_search_with_score(question, k=top_k)

            results = []
            for doc, score in docs_with_scores:
                print(score)
                # similarity = min(0.0, score)  # Convert distance → similarity
                results.append({
                    "id": doc.metadata.get("chunk_index", "unknown"),
                    "score": score,  # ✅ 0.723, 0.891, 0.654, etc.
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
            return results
        except Exception as e:
            print(f"Retrieval error: {e}")
            return []
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def ask(self, question: str, document_id: str) -> Dict[str, Any]:
        contexts = self.retrieve(question, document_id)
        print(contexts)
        if not contexts or contexts[0]['score'] > 1.5:  # Guardrail
            return {
                "answer": "Not found in document",
                "sources": [],
                "confidence_score": 1.5,
                "confidence_reason": f"Top similarity {contexts[0]['score'] if contexts else 0:.2f} > 1.5",
                "is_reliable": False
            }
        
        contexts.sort(key=lambda x: x['id'])
        context_text = "\n---\n".join([ctx['content'] for ctx in contexts])
        
        answer = self.chain.invoke({"context": context_text, "question": question}).strip()
        print(answer)
        if "Not found in document" in answer:
            return {"answer": answer, "confidence_score": 0, "is_reliable": False,"sources":[],"confidence_reason":"None"}
        
        confidence = self.calculate_confidence(question, contexts, answer)

        print(answer)
        return {
            "answer": answer,
            "sources": contexts,
            "confidence_score": confidence["score"],
            "confidence_reason": confidence["reason"],
            "is_reliable": confidence["score"] < 1.5
        }
    
    def calculate_confidence(self, question: str, contexts: List[Dict], answer: str) -> Dict:
        avg_similarity = np.mean([ctx['score'] for ctx in contexts[:3]])
        q_words = set(question.lower().split())
        a_words = set(answer.lower().split())
        keyword_overlap = len(q_words.intersection(a_words)) / len(q_words) if q_words else 0
        
        final_score = avg_similarity * 0.6 + keyword_overlap * 0.4
        return {
            "score": round(min(final_score, 1.0), 3),
            "reason": f"Sim:{avg_similarity:.2f}, KW:{keyword_overlap:.2f}"
        }
