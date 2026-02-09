# import os
# from google import genai
# from google.genai import types
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma
# import numpy as np
# from typing import List, Dict, Any
# from pathlib import Path
# from tenacity import (
#     retry,
#     stop_after_attempt,
#     wait_exponential,
#     retry_if_exception_type
# )
# from dotenv import load_dotenv
# load_dotenv() 

# class RAGPipeline:
#     def __init__(self):
#         self.client = genai.Client()
#         self.embeddings = HuggingFaceEmbeddings(
#             model_name="sentence-transformers/all-mpnet-base-v2",
#             model_kwargs={'device': 'cpu'}
#         )
#         self.persist_directory = Path("chroma_db")
#         self.collection_prefix = "logistics_docs"
#         # Updated to the correct model identifier
#         self.model_name = "gemini-3-flash-preview" 
    
#     def retrieve(self, question: str, document_id: str, top_k: int = 6) -> List[Dict]:
#         try:
#             collection_name = f"{self.collection_prefix}_{document_id}"
#             vectorstore = Chroma(
#                 collection_name=collection_name,
#                 embedding_function=self.embeddings,
#                 persist_directory=str(self.persist_directory)
#             )
            
#             # Use MMR (Max Marginal Relevance) to get diverse chunks 
#             # This prevents getting 3 chunks that all say the same thing
#             docs_with_scores = vectorstore.max_marginal_relevance_search(
#                 question, k=top_k, fetch_k=10
#             )
            
#             # Since MMR doesn't return scores directly in the same way, 
#             # we'll map them. For logistics, simple similarity is often fine too.
#             results = []
#             for i, doc in enumerate(docs_with_scores):
#                 results.append({
#                     "id": doc.metadata.get("chunk_index", i),
#                     "score": 0.9, # Placeholder for MMR
#                     "content": doc.page_content,
#                     "metadata": doc.metadata
#                 })
#             return results
#         except Exception as e:
#             print(f"Retrieval error: {e}")
#             return []

#     def ask(self, question: str, document_id: str) -> Dict[str, Any]:
#         contexts = self.retrieve(question, document_id)
#         if not contexts:
#             return {"answer": "Not found in document", "is_reliable": False}
        
#         # Sort contexts by chunk_index to maintain document flow
#         # This is CRITICAL for logistics docs where the table header 
#         # is in chunk N and the data is in chunk N+1.
#         contexts.sort(key=lambda x: x['id'])
#         # contexts = contexts[:1]
#         context_text = "\n---\n".join([ctx['content'] for ctx in contexts])
        
#         answer = self.generate_with_gemini(context_text, question)
        
#         # If Gemini says not found, we don't calculate confidence
#         if "Not found" in answer:
#             return {"answer": answer, "confidence_score": 0, "is_reliable": False}

#         confidence = self.calculate_confidence(question, contexts, answer)
        
#         return {
#             "answer": answer,
#             "sources": contexts[:3],
#             "confidence_score": confidence["score"],
#             "confidence_reason": confidence["reason"],
#             "is_reliable": confidence["score"] > 0.6
#         }
#     @retry(
#         stop=stop_after_attempt(3),
#         wait=wait_exponential(multiplier=1, min=2, max=10),
#         retry=retry_if_exception_type(Exception) 
#     )
#     def generate_with_gemini(self, context: str, question: str) -> str:
#         try:
#             # Refined prompt for better grounding
#             system_prompt = (
#                 "You are a specialized logistics document assistant. "
#                 "Your task is to answer user queries using ONLY the provided context snippets. "
#                 "If the specific information (like weight, price, or location) is not in the context, "
#                 "respond exactly with: 'Not found in document'. "
#                 "Do not use external knowledge.\n\n"
#                 f"### CONTEXT START ###\n{context}\n### CONTEXT END ###"
#             )

#             response = self.client.models.generate_content(
#                 model=self.model_name,
#                 contents=question,
#                 config=types.GenerateContentConfig(
#                     system_instruction=system_prompt,
#                     temperature=0.0, # Lower temperature for higher factual accuracy
#                     max_output_tokens=512,
#                     # Optional: use safety settings to prevent hallucinations
#                 )
#             )

#             return response.text.strip()
#         except Exception as e:
#             raise e
    
    
#     def calculate_confidence(self, question: str, contexts: List[Dict], answer: str) -> Dict:
#         if not contexts or "Not found in document" in answer:
#             return {"score": 0.0, "reason": "No relevant context or info missing"}
        
#         # Calculate semantic grounding (context similarity)
#         avg_similarity = np.mean([ctx['score'] for ctx in contexts[:3]])
        
#         # Check if key entities from the question appear in the answer/context
#         # This is a simple heuristic; for production, consider a small LLM evaluator
#         final_score = round(min(avg_similarity, 1.0), 3)
        
#         return {
#             "score": final_score,
#             "reason": f"Retrieval Confidence: {final_score:.2f}"
#         }

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
            ("system", """You are a logistics document assistant. Answer using ONLY the context provided.
If the exact information is not in the context, respond exactly: "Not found in document".

Context:
{context}"""),
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
        if not contexts or contexts[0]['score'] > 1.0:  # Guardrail
            return {
                "answer": "Not found in document",
                "sources": [],
                "confidence_score": 1.0,
                "confidence_reason": f"Top similarity {contexts[0]['score'] if contexts else 0:.2f} > 1.0",
                "is_reliable": False
            }
        
        contexts.sort(key=lambda x: x['id'])
        context_text = "\n---\n".join([ctx['content'] for ctx in contexts[:3]])
        
        answer = self.chain.invoke({"context": context_text, "question": question}).strip()
        
        if "Not found in document" in answer:
            return {"answer": answer, "confidence_score": 0, "is_reliable": False}
        
        confidence = self.calculate_confidence(question, contexts, answer)
        return {
            "answer": answer,
            "sources": contexts[:3],
            "confidence_score": confidence["score"],
            "confidence_reason": confidence["reason"],
            "is_reliable": confidence["score"] < 1.0
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
