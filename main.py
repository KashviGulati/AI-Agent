"""
main.py - RAG Query System and FastAPI Server with Data Analysis
Handles querying, AI responses, API endpoints, and comprehensive data analysis
"""

import os
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

# FastAPI and web
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# ChromaDB and AI
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

# Import our systems
from ingest import DocumentIngestion
from data_analysis import DataAnalysisSystem

# ---------- CONFIG ----------
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "document_store"
EMBED_MODEL = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-1.5-flash"
TOP_K = 10
RELEVANCE_THRESHOLD = 1.2  # Distance threshold for relevance
# ----------------------------

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment. Add it to .env file.")
genai.configure(api_key=api_key)


class QuerySystem:
    """Advanced RAG query system with AI agent capabilities"""

    def __init__(self):
        # Initialize ChromaDB connection
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        try:
            self.collection = self.client.get_collection(name=COLLECTION_NAME)
        except Exception:
            self.collection = self.client.create_collection(name=COLLECTION_NAME)

        # Initialize ingestion system (for file management)
        self.ingestion = DocumentIngestion()

        # Lazy load models
        self.embedder = None
        self.gemini_model = None

    def _load_models(self):
        """Load AI models lazily"""
        if self.embedder is None:
            print("Loading embedding model for queries...")
            self.embedder = SentenceTransformer(EMBED_MODEL)

        if self.gemini_model is None:
            print("Loading Gemini model...")
            self.gemini_model = genai.GenerativeModel(GEMINI_MODEL)

    def _create_enhanced_prompt(self, query: str, context: str, context_metadata: List[Dict], low_relevance=False) -> str:
        """Create an enhanced prompt for the AI agent"""
        # Analyze context sources
        source_info = {}
        for meta in context_metadata:
            filename = meta.get("filename", "Unknown")
            if filename not in source_info:
                source_info[filename] = {"chunks": 0, "total_chunks": meta.get("total_chunks", 0)}
            source_info[filename]["chunks"] += 1

        source_summary = "\n".join([
            f"- {filename}: {info['chunks']} relevant sections (out of {info['total_chunks']} total)"
            for filename, info in source_info.items()
        ])

        relevance_note = ""
        if low_relevance:
            relevance_note = (
                "\n⚠️ NOTE: The provided context may not directly answer the question, "
                "but it is the closest information found in the document. Use it as background "
                "and combine it with your own knowledge to answer."
            )

        return f"""You are an expert AI assistant with access to a comprehensive knowledge base. 
Your role is to provide accurate, detailed, and actionable responses based on the provided context.

KNOWLEDGE BASE SOURCES:
{source_summary}
{relevance_note}

ANALYSIS INSTRUCTIONS:
1. Carefully read and analyze ALL provided context chunks
2. Synthesize information from multiple sources when relevant
3. If context does not fully answer the query, use your own reasoning and knowledge to provide a relevant, accurate answer
4. Clearly distinguish between information from the document and your own inferred knowledge
5. Structure your response for maximum clarity and usefulness

USER QUERY: {query}

RELEVANT CONTEXT:
{context}

Your detailed response:"""

    def _calculate_confidence(self, distances: List[float], num_chunks: int) -> float:
        """Calculate confidence score based on relevance and coverage"""
        if not distances:
            return 0.0

        avg_relevance = sum(1 - d for d in distances) / len(distances)
        coverage_factor = min(num_chunks / 5, 1.0)  # Cap at 5 chunks

        confidence = (avg_relevance * 0.7) + (coverage_factor * 0.3)
        return min(confidence, 1.0)

    def query(self, query: str, top_k: int = TOP_K, include_debug: bool = False) -> Dict:
        """Main query method with enhanced AI agent response"""
        print(f"\n=== Processing Query ===")
        print(f"Query: {query}")
        print(f"Retrieving top {top_k} chunks...")

        self._load_models()

        query_embedding = self.embedder.encode(query).tolist()

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            return {
                "answer": f"Error searching knowledge base: {str(e)}",
                "sources": [],
                "confidence": 0.0,
                "status": "error"
            }

        if not results["documents"] or not results["documents"][0]:
            return {
                "answer": "No documents found in the knowledge base. Please upload some documents first.",
                "sources": [],
                "confidence": 0.0,
                "status": "no_results"
            }

        docs = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        # First pass: strict relevance filter
        relevant_docs = []
        relevant_metadata = []
        relevant_distances = []

        for doc, meta, dist in zip(docs, metadatas, distances):
            if dist <= RELEVANCE_THRESHOLD:
                relevant_docs.append(doc)
                relevant_metadata.append(meta)
                relevant_distances.append(dist)

        low_relevance_mode = False
        if not relevant_docs:
            # Fallback: take top_k anyway, but mark as low relevance
            relevant_docs = docs
            relevant_metadata = metadatas
            relevant_distances = distances
            low_relevance_mode = True

        print(f"Found {len(relevant_docs)} chunks (low_relevance_mode={low_relevance_mode})")

        # Prepare sources
        sources = [
            {
                "filename": meta.get("filename", "Unknown"),
                "chunk_index": meta.get("chunk_index", 0),
                "total_chunks": meta.get("total_chunks", 0),
                "relevance_score": round(1 - dist, 3),
                "upload_time": meta.get("upload_time", "Unknown")
            }
            for meta, dist in zip(relevant_metadata, relevant_distances)
        ]

        # Create context
        context = "\n\n---DOCUMENT CHUNK---\n\n".join(relevant_docs)

        # Build prompt
        prompt = self._create_enhanced_prompt(query, context, relevant_metadata, low_relevance=low_relevance_mode)

        # Get AI answer
        try:
            print("Generating AI response...")
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=2048,
                )
            )
            answer = response.text.strip() if response and hasattr(response, "text") else "No response generated"
        except Exception as e:
            print(f"Error generating AI response: {e}")
            answer = f"Error generating response: {str(e)}"

        # Confidence score
        confidence = self._calculate_confidence(relevant_distances, len(relevant_docs))

        result = {
            "answer": answer,
            "sources": sources,
            "confidence": round(confidence, 3),
            "total_chunks_found": len(relevant_docs),
            "query_processed": query,
            "status": "success"
        }

        if include_debug:
            result["debug"] = {
                "all_distances": distances,
                "relevance_threshold": RELEVANCE_THRESHOLD,
                "chunks_before_filtering": len(docs),
                "chunks_after_filtering": len(relevant_docs),
                "low_relevance_mode": low_relevance_mode,
                "context_preview": context[:500] + "..." if len(context) > 500 else context
            }

        print(f"✅ Query processed successfully")
        print(f"   - Confidence: {confidence:.3f}")
        print(f"   - Sources: {len(sources)}")
        print(f"   - Answer length: {len(answer)} characters")

        return result

    def get_query_stats(self) -> Dict:
        """Get query system statistics"""
        try:
            collection_count = self.collection.count()
        except:
            collection_count = 0
        
        return {
            "total_chunks_in_db": collection_count,
            "embedding_model": EMBED_MODEL,
            "ai_model": GEMINI_MODEL,
            "top_k": TOP_K,
            "relevance_threshold": RELEVANCE_THRESHOLD
        }


# === FastAPI setup ===
print("Initializing Enhanced RAG System with Data Analysis...")
query_system = QuerySystem()
data_analysis_system = DataAnalysisSystem()
print("Systems initialized successfully!")

app = FastAPI(
    title="Enhanced RAG System with Data Analysis",
    description="Document ingestion, intelligent querying, and comprehensive data analysis system",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    top_k: int = TOP_K
    include_debug: bool = False

class FileRemoveRequest(BaseModel):
    file_id: str

class BulkIngestRequest(BaseModel):
    directory_path: str

class MLTrainingRequest(BaseModel):
    file_id: str
    target_column: str
    task_type: str = "auto"  # "auto", "classification", "regression"

class ClusteringRequest(BaseModel):
    file_id: str
    n_clusters: Optional[int] = None

class GANVisualizationRequest(BaseModel):
    file_id: str
    columns: Optional[List[str]] = None

class AdvancedVisualizationRequest(BaseModel):
    file_id: str
    chart_types: Optional[List[str]] = None


# === RAG Endpoints ===
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    allowed_extensions = {'.pdf', '.docx', '.doc', '.pptx', '.ppt', '.txt', '.csv', '.xlsx', '.xls'}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file_ext}'. Allowed: {', '.join(allowed_extensions)}"
        )

    try:
        print(f"\n📄 Received upload request for: {file.filename}")
        content = await file.read()
        
        # Standard document ingestion for text-based files
        if file_ext in {'.pdf', '.docx', '.doc', '.pptx', '.ppt', '.txt'}:
            result = query_system.ingestion.ingest_document(content, file.filename)
        else:
            # For CSV/Excel, only add to file manager without text extraction
            result = query_system.ingestion.file_manager.add_file(content, file.filename)
            
            # Register for data analysis if it's a data file
            if file_ext in {'.csv', '.xlsx', '.xls'} and result["status"] in ["added", "duplicate"]:
                file_id = result["file_id"]
                file_path = query_system.ingestion.file_manager.get_file_info(file_id)["file_path"]
                data_result = data_analysis_system.register_data_file(file_id, file_path)
                result["data_analysis_registration"] = data_result
        
        return JSONResponse(content=result)
    except Exception as e:
        print(f"❌ Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/query")
async def query_documents(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        print(f"\n🤖 Received query: {request.query[:100]}...")
        result = query_system.query(request.query, request.top_k, request.include_debug)
        return JSONResponse(content=result)
    except Exception as e:
        print(f"❌ Query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/files")
async def list_files():
    try:
        result = query_system.ingestion.list_files()
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"List files failed: {str(e)}")


@app.delete("/remove")
async def remove_file(request: FileRemoveRequest):
    if not request.file_id:
        raise HTTPException(status_code=400, detail="File ID is required")

    try:
        print(f"\n🗑️ Removing file: {request.file_id}")
        result = query_system.ingestion.remove_document(request.file_id)
        
        # Also cleanup data analysis if it exists
        data_analysis_system.cleanup_analysis(request.file_id)
        
        return JSONResponse(content=result)
    except Exception as e:
        print(f"❌ Remove failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Remove failed: {str(e)}")


@app.post("/bulk-ingest")
async def bulk_ingest(request: BulkIngestRequest):
    if not request.directory_path:
        raise HTTPException(status_code=400, detail="Directory path is required")

    directory_path = Path(request.directory_path)
    if not directory_path.exists():
        raise HTTPException(status_code=400, detail="Directory does not exist")

    try:
        print(f"\n📂 Starting bulk ingestion from: {directory_path}")
        result = query_system.ingestion.bulk_ingest_from_directory(str(directory_path))
        return JSONResponse(content=result)
    except Exception as e:
        print(f"❌ Bulk ingest failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Bulk ingest failed: {str(e)}")


# === Data Analysis Endpoints ===
@app.post("/data-analysis/eda")
async def perform_eda(file_id: str):
    """Perform Exploratory Data Analysis on uploaded CSV/Excel file"""
    try:
        print(f"\n📊 Performing EDA for file: {file_id}")
        result = data_analysis_system.perform_eda(file_id)
        return JSONResponse(content=result)
    except Exception as e:
        print(f"❌ EDA failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"EDA failed: {str(e)}")


@app.post("/data-analysis/ml-training")
async def train_ml_models(request: MLTrainingRequest):
    """Train machine learning models on the dataset"""
    try:
        print(f"\n🤖 Training ML models for file: {request.file_id}, target: {request.target_column}")
        result = data_analysis_system.train_ml_models(
            request.file_id, 
            request.target_column, 
            request.task_type
        )
        return JSONResponse(content=result)
    except Exception as e:
        print(f"❌ ML training failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"ML training failed: {str(e)}")


@app.post("/data-analysis/clustering")
async def perform_clustering(request: ClusteringRequest):
    """Perform clustering analysis on the dataset"""
    try:
        print(f"\n🎯 Performing clustering for file: {request.file_id}")
        result = data_analysis_system.perform_clustering(request.file_id, request.n_clusters)
        return JSONResponse(content=result)
    except Exception as e:
        print(f"❌ Clustering failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Clustering failed: {str(e)}")


@app.post("/data-analysis/gan-visualization")
async def generate_gan_visualization(request: GANVisualizationRequest):
    """Generate GAN-based data visualization"""
    try:
        print(f"\n🎨 Generating GAN visualization for file: {request.file_id}")
        result = data_analysis_system.generate_gan_visualization(request.file_id, request.columns)
        return JSONResponse(content=result)
    except Exception as e:
        print(f"❌ GAN visualization failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"GAN visualization failed: {str(e)}")


@app.post("/data-analysis/advanced-visualizations")
async def create_advanced_visualizations(request: AdvancedVisualizationRequest):
    """Create advanced data visualizations"""
    try:
        print(f"\n📈 Creating advanced visualizations for file: {request.file_id}")
        result = data_analysis_system.create_advanced_visualizations(request.file_id, request.chart_types)
        return JSONResponse(content=result)
    except Exception as e:
        print(f"❌ Advanced visualization failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Advanced visualization failed: {str(e)}")


@app.get("/data-analysis/insights/{file_id}")
async def get_data_insights(file_id: str):
    """Get AI-powered insights about the dataset"""
    try:
        print(f"\n🧠 Generating insights for file: {file_id}")
        result = data_analysis_system.get_data_insights(file_id)
        return JSONResponse(content=result)
    except Exception as e:
        print(f"❌ Insights generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Insights generation failed: {str(e)}")


@app.get("/data-analysis/history")
async def get_analysis_history(file_id: str = None):
    """Get history of all analyses performed"""
    try:
        result = data_analysis_system.get_analysis_history(file_id)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get analysis history: {str(e)}")


@app.get("/data-analysis/export/{analysis_key}")
async def export_analysis_results(analysis_key: str, format_type: str = "json"):
    """Export analysis results in specified format"""
    try:
        result = data_analysis_system.export_results(analysis_key, format_type)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@app.delete("/data-analysis/cleanup/{file_id}")
async def cleanup_data_analysis(file_id: str):
    """Clean up analysis data for a specific file"""
    try:
        result = data_analysis_system.cleanup_analysis(file_id)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")


# === System Endpoints ===
@app.get("/stats")
async def get_stats():
    try:
        ingestion_stats = query_system.ingestion.get_ingestion_stats()
        query_stats = query_system.get_query_stats()
        
        # Get data analysis stats
        data_analysis_stats = {
            "registered_data_files": len(data_analysis_system.data_files),
            "total_analyses_performed": len(data_analysis_system.analysis_results),
            "available_analysis_types": [
                "EDA", "ML Training", "Clustering", "GAN Visualization", 
                "Advanced Visualizations", "Data Insights"
            ]
        }

        combined_stats = {
            "system_status": "operational",
            "timestamp": datetime.now().isoformat(),
            "ingestion": ingestion_stats,
            "querying": query_stats,
            "data_analysis": data_analysis_stats
        }
        return JSONResponse(content=combined_stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats failed: {str(e)}")


@app.get("/health")
async def health_check():
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {}
    }

    try:
        query_system.collection.count()
        health_status["components"]["chromadb"] = "healthy"
    except Exception as e:
        health_status["components"]["chromadb"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"

    try:
        query_system._load_models()
        health_status["components"]["ai_models"] = "healthy"
    except Exception as e:
        health_status["components"]["ai_models"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"

    try:
        # Test data analysis system
        len(data_analysis_system.data_files)
        health_status["components"]["data_analysis"] = "healthy"
    except Exception as e:
        health_status["components"]["data_analysis"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"

    return JSONResponse(content=health_status)


# === Enhanced CLI Interface ===
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced RAG System with Data Analysis")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--query", type=str, help="Run a single query from command line")
    parser.add_argument("--interactive", action="store_true", help="Start interactive query mode")
    parser.add_argument("--analyze-data", type=str, help="Perform data analysis on a CSV/Excel file")
    parser.add_argument("--data-insights", type=str, help="Get insights for a registered data file")

    args = parser.parse_args()

    if args.query:
        print(f"\n🤖 Processing query: {args.query}")
        result = query_system.query(args.query)
        print(f"\n📋 Answer:\n{result['answer']}")
        print(f"\n📚 Sources: {len(result['sources'])} documents")
        print(f"🎯 Confidence: {result['confidence']:.3f}")

    elif args.interactive:
        print("\n🤖 Interactive RAG Query Mode")
        print("Type your questions (or 'quit' to exit):")
        print("-" * 40)

        while True:
            try:
                q = input("\n❓ Your question: ").strip()
                if q.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                if not q:
                    continue
                print("🔍 Searching knowledge base...")
                res = query_system.query(q)
                print(f"\n📋 Answer:\n{res['answer']}")
                print(f"📊 Confidence: {res['confidence']:.3f}")
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")

    elif args.analyze_data:
        print(f"\n📊 Analyzing data file: {args.analyze_data}")
        try:
            # Register the file
            file_path = Path(args.analyze_data)
            if not file_path.exists():
                print("❌ File not found!")
            else:
                file_id = f"cli_{file_path.stem}"
                register_result = data_analysis_system.register_data_file(file_id, str(file_path))
                if register_result["status"] == "success":
                    print("✅ File registered successfully")
                    
                    # Perform EDA
                    print("🔍 Performing EDA...")
                    eda_result = data_analysis_system.perform_eda(file_id)
                    if eda_result["status"] == "success":
                        stats = eda_result["summary_statistics"]["dataset_info"]
                        print(f"📊 Dataset: {stats['shape'][0]} rows, {stats['shape'][1]} columns")
                        print(f"💾 Memory usage: {stats['memory_usage_mb']} MB")
                        print("✅ EDA completed successfully")
                    else:
                        print(f"❌ EDA failed: {eda_result.get('message', 'Unknown error')}")
                else:
                    print(f"❌ File registration failed: {register_result.get('message', 'Unknown error')}")
        except Exception as e:
            print(f"❌ Error: {str(e)}")

    elif args.data_insights:
        print(f"\n🧠 Generating insights for file: {args.data_insights}")
        try:
            result = data_analysis_system.get_data_insights(args.data_insights)
            if result["status"] == "success":
                insights = result["insights"]
                print(f"📊 Dataset Overview:")
                print(f"   - Rows: {insights['dataset_overview']['total_rows']}")
                print(f"   - Columns: {insights['dataset_overview']['total_columns']}")
                print(f"   - Data Quality Score: {insights['data_quality']['completeness_score']:.1f}%")
                print(f"🔍 Recommendations: {len(insights['recommendations'])}")
                for i, rec in enumerate(insights['recommendations'][:3], 1):
                    print(f"   {i}. {rec}")
            else:
                print(f"❌ Insights generation failed: {result.get('message', 'Unknown error')}")
        except Exception as e:
            print(f"❌ Error: {str(e)}")

    else:
        print("\n🚀 Starting Enhanced RAG System Server...")
        print(f"📊 Data Analysis Features: Enabled")
        print(f"🤖 AI Agent: Enabled")
        print(f"🔍 RAG System: Enabled")
        uvicorn.run("main:app", host=args.host, port=args.port, reload=args.reload, log_level="info")