"""
config.py - Central configuration for the Day 1 Document Ingestion Pipeline.
"""

from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "Data"
CHROMA_DIR = ROOT_DIR / "chroma_db"
COLLECTION_NAME = "clinical_guidelines"

# Chunking (in tokens; approx 4 chars per token)
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# Embedding model (local, no API key needed)
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
