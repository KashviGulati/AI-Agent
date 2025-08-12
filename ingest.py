# ingest.py
import os
import glob
import uuid
from datetime import datetime
from sentence_transformers import SentenceTransformer
import chromadb
from pypdf import PdfReader
from docx import Document  # for .docx support

# ---------- CONFIG ----------
DATA_DIR = "./data"
CHROMA_DIR = "./chroma_db"
CHUNK_SIZE = 1000  # chars per chunk
CHUNK_OVERLAP = 200
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "memories"
# ----------------------------

# Ensure data folder exists
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

print("Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL)

print("Starting Chroma client...")
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

def clean_text(s: str) -> str:
    return " ".join(s.split())

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

def extract_text_from_pdf(path):
    reader = PdfReader(path)
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception as e:
            print("PDF page read error:", e)
            pages.append("")
    return "\n".join(pages)

def extract_text_from_docx(path):
    doc = Document(path)
    return "\n".join([para.text for para in doc.paragraphs])

def ingest_folder(folder_path=DATA_DIR):
    ids, documents, metadatas = [], [], []

    files = glob.glob(os.path.join(folder_path, "*"))
    if not files:
        print(f"No files found in {folder_path}. Add PDFs, TXT, or DOCX files and try again.")
        return

    for fpath in files:
        fname = os.path.basename(fpath)
        ext = fname.split(".")[-1].lower()
        print(f"Processing: {fname}")

        if ext == "pdf":
            text = extract_text_from_pdf(fpath)
        elif ext in ("txt", "md"):
            with open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
                text = fh.read()
        elif ext == "docx":
            text = extract_text_from_docx(fpath)
        else:
            print(f"Skipping unsupported file: {fpath}")
            continue

        text = clean_text(text)
        chunks = chunk_text(text)
        for i, c in enumerate(chunks):
            doc_id = f"{fname}_{i}_{uuid.uuid4().hex[:6]}"
            ids.append(doc_id)
            documents.append(c)
            metadatas.append({
                "source": fname,
                "source_path": fpath,
                "chunk_index": i,
                "ingested_at": datetime.utcnow().isoformat()
            })

    if not documents:
        print("No valid content to ingest.")
        return

    print("Creating embeddings (this may take a while)...")
    embs = embedder.encode(documents, show_progress_bar=True, convert_to_numpy=True)

    print(f"Adding {len(documents)} chunks to Chroma...")
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embs.tolist()
    )
    print(f"✅ Ingested {len(documents)} chunks into Chroma at {CHROMA_DIR}.")

if __name__ == "__main__":
    ingest_folder()
