"""
main.py - RAG Query System and FastAPI Server with Data Analysis
"""

import os
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv

from ingest import DocumentIngestion
from data_analysis import DataAnalysisSystem

# Config
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "document_store"
EMBED_MODEL = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-2.0-flash-exp"
TOP_K = 10
RELEVANCE_THRESHOLD = 1.2

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment")
genai.configure(api_key=api_key)


class QuerySystem:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        try:
            self.collection = self.client.get_collection(name=COLLECTION_NAME)
        except:
            self.collection = self.client.create_collection(name=COLLECTION_NAME)
        self.ingestion = DocumentIngestion()
        self.embedder = None
        self.gemini_model = None

    def _load_models(self):
        if self.embedder is None:
            self.embedder = SentenceTransformer(EMBED_MODEL)
        if self.gemini_model is None:
            self.gemini_model = genai.GenerativeModel(
                GEMINI_MODEL,
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            )

    def query(self, query: str, top_k: int = TOP_K, include_debug: bool = False) -> Dict:
        self._load_models()
        query_embedding = self.embedder.encode(query).tolist()

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            return {"answer": f"Error: {str(e)}", "sources": [], "confidence": 0.0, "status": "error"}

        if not results["documents"] or not results["documents"][0]:
            return {"answer": "No documents found.", "sources": [], "confidence": 0.0, "status": "no_results"}

        docs = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        relevant_docs = [doc for doc, dist in zip(docs, distances) if dist <= RELEVANCE_THRESHOLD]
        relevant_metadata = [meta for meta, dist in zip(metadatas, distances) if dist <= RELEVANCE_THRESHOLD]
        relevant_distances = [dist for dist in distances if dist <= RELEVANCE_THRESHOLD]

        if not relevant_docs:
            relevant_docs, relevant_metadata, relevant_distances = docs, metadatas, distances

        sources = [
            {
                "filename": meta.get("filename", "Unknown"),
                "chunk_index": meta.get("chunk_index", 0),
                "relevance_score": round(1 - dist, 3)
            }
            for meta, dist in zip(relevant_metadata, relevant_distances)
        ]

        context = "\n\n---\n\n".join(relevant_docs)
        prompt = f"""You are an AI assistant. Answer based on the context provided.

Query: {query}

Context:
{context}

Response:"""

        try:
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(temperature=0.3, max_output_tokens=2048)
            )
            answer = response.text.strip() if response and hasattr(response, "text") else "No response"
        except Exception as e:
            answer = f"Error: {str(e)}"

        confidence = sum(1 - d for d in relevant_distances) / len(relevant_distances) if relevant_distances else 0

        return {
            "answer": answer,
            "sources": sources,
            "confidence": round(confidence, 3),
            "total_chunks_found": len(relevant_docs),
            "status": "success"
        }


# FastAPI Setup
query_system = QuerySystem()
data_analysis_system = DataAnalysisSystem()

app = FastAPI(title="RAG System", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


# Pydantic Models
class QueryRequest(BaseModel):
    query: str
    top_k: int = TOP_K

class FileRemoveRequest(BaseModel):
    file_id: str

class MLTrainingRequest(BaseModel):
    file_id: str
    target_column: str
    task_type: str = "auto"

class ClusteringRequest(BaseModel):
    file_id: str
    n_clusters: Optional[int] = None


# Endpoints
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "No filename")

    file_ext = Path(file.filename).suffix.lower()
    allowed = {'.pdf', '.docx', '.doc', '.pptx', '.ppt', '.txt', '.csv', '.xlsx', '.xls'}
    if file_ext not in allowed:
        raise HTTPException(400, f"Unsupported type: {file_ext}")

    try:
        content = await file.read()
        if file_ext in {'.pdf', '.docx', '.doc', '.pptx', '.ppt', '.txt'}:
            result = query_system.ingestion.ingest_document(content, file.filename)
        else:
            result = query_system.ingestion.file_manager.add_file(content, file.filename)
            if file_ext in {'.csv', '.xlsx', '.xls'} and result["status"] in ["added", "duplicate"]:
                file_info = query_system.ingestion.file_manager.get_file_info(result["file_id"])
                if file_info:
                    data_analysis_system.register_data_file(result["file_id"], file_info["file_path"])
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/query")
async def query_documents(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(400, "Empty query")
    try:
        result = query_system.query(request.query, request.top_k)
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/files")
async def list_files():
    return JSONResponse(query_system.ingestion.list_files())


@app.delete("/remove")
async def remove_file(request: FileRemoveRequest):
    result = query_system.ingestion.remove_document(request.file_id)
    data_analysis_system.cleanup_analysis(request.file_id)
    return JSONResponse(result)


@app.get("/data-analysis/eda")
async def perform_eda(file_id: str):
    if file_id not in data_analysis_system.data_files:
        raise HTTPException(404, "File not registered")
    return JSONResponse(data_analysis_system.perform_eda(file_id))


@app.post("/data-analysis/ml-training")
async def train_ml_models(request: MLTrainingRequest):
    result = data_analysis_system.train_ml_models(request.file_id, request.target_column, request.task_type)
    return JSONResponse(result)


@app.post("/data-analysis/clustering")
async def perform_clustering(request: ClusteringRequest):
    result = data_analysis_system.perform_clustering(request.file_id, request.n_clusters)
    return JSONResponse(result)


@app.get("/data-analysis/insights/{file_id}")
async def get_insights(file_id: str):
    return JSONResponse(data_analysis_system.get_data_insights(file_id))


@app.get("/stats")
async def get_stats():
    return JSONResponse({
        "timestamp": datetime.now().isoformat(),
        "ingestion": query_system.ingestion.get_ingestion_stats(),
        "total_chunks": query_system.collection.count(),
        "registered_files": len(data_analysis_system.data_files)
    })


@app.get("/health")
async def health():
    return JSONResponse({"status": "healthy", "timestamp": datetime.now().isoformat()})


# CLI
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["server", "cli"], default="server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev only)")
    args = parser.parse_args()

    if args.mode == "server":
        print(f"Starting server on {args.host}:{args.port}")
        # Fix for Windows multiprocessing with reload
        if args.reload:
            uvicorn.run("main:app", host=args.host, port=args.port, reload=True, reload_dirs=["."])
        else:
            uvicorn.run(app, host=args.host, port=args.port)
    else:
        system = QuerySystem()
        print("RAG System CLI - Type 'exit' to quit")
        while True:
            try:
                user_input = input("\nQuery> ").strip()
                if user_input.lower() in ["exit", "quit"]:
                    break
                if user_input:
                    result = system.query(user_input)
                    print(f"\n{result['answer']}\n")
                    print(f"Confidence: {result['confidence']:.2%} | Sources: {result['total_chunks_found']}")
            except KeyboardInterrupt:
                break