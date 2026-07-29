"""Central config: paths, model IDs, and pipeline knobs."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
TEXT_DIR = DATA_DIR / "text_files"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

VECTOR_COLLECTION_NAME = "pdf_documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 1024

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

RETRIEVAL_TOP_K = 5
RETRIEVAL_SCORE_THRESHOLD = 0.2

# Bounds the retrieve -> grade -> rewrite loop in agent.py.
MAX_QUERY_REWRITES = 2
