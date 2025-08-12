# main.py
import chromadb
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------- CONFIG ----------
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "memories"
EMBED_MODEL = "all-MiniLM-L6-v2"
SUM_MODEL = "t5-small"  # small and safe on CPU
TOP_K = 5
# ----------------------------

print("Starting Chroma client...")
client = chromadb.PersistentClient(path=CHROMA_DIR)

# Create or load the collection
try:
    collection = client.get_collection(name=COLLECTION_NAME)
except Exception:
    collection = client.create_collection(name=COLLECTION_NAME)

print("Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL)

print("Loading summarization model (this may take a moment)...")
tokenizer = AutoTokenizer.from_pretrained(SUM_MODEL)
summ_model = AutoModelForSeq2SeqLM.from_pretrained(SUM_MODEL)

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    top_k: int = TOP_K

def make_prompt(query, docs):
    context = "\n\n---\n\n".join(docs)
    prompt = (
        f"Your job: based on the context below, answer the question, extract commitments, deadlines, "
        f"and people mentioned.\n\nQuestion: {query}\n\nContext:\n{context}\n\n"
        "Return a short answer and then list commitments, deadlines and people in a readable form."
    )
    return prompt

def summarize_text(prompt, max_tokens=150):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = summ_model.generate(**inputs, max_new_tokens=max_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.post("/query")
def query(req: QueryRequest):
    emb = embedder.encode(req.query).tolist()
    results = collection.query(
        query_embeddings=[emb],
        n_results=req.top_k,
        include=["documents", "metadatas", "distances"]
    )

    docs = results["documents"][0] if results["documents"] else []
    metas = results["metadatas"][0] if results["metadatas"] else []

    combined_prompt = make_prompt(req.query, docs)
    answer = summarize_text(combined_prompt)

    # return answer plus sources for traceability
    sources = []
    for m in metas:
        sources.append({"source": m.get("source"), "chunk_index": m.get("chunk_index")})

    return {"answer": answer, "sources": sources}
