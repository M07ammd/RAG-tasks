from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "Data"
CHROMA_DIR = ROOT_DIR / "chroma_db"
COLLECTION_NAME = "clinical_guidelines"

# Chunking 
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# Embedding model 
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
