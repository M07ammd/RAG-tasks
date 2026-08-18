from pathlib import Path
import os

# Paths
ROOT_DIR = Path(__file__).parent


def _load_env_file() -> None:
	"""Load simple KEY=VALUE settings without adding a dependency."""
	env_path = ROOT_DIR / ".env"
	if not env_path.exists():
		return
	for raw_line in env_path.read_text(encoding="utf-8").splitlines():
		line = raw_line.strip()
		if not line or line.startswith("#") or "=" not in line:
			continue
		key, value = line.split("=", 1)
		os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env_file()
DATA_DIR = ROOT_DIR / "Data"
CHROMA_DIR = ROOT_DIR / "chroma_db"
COLLECTION_NAME = "clinical_guidelines"

# Chunking 
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# Embedding model 
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Grounded generation. The default follows the reference notebook's illustrative
# cutoff; calibrate it against the Task 2 evaluation scores when those are available.
GROUNDING_THRESHOLD = float(os.getenv("GROUNDING_THRESHOLD", "0.30"))
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

# Local LLM configuration. Ollama serves the model on localhost by default.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://127.0.0.1:11434")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "llama3.2:3b")
