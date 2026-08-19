import hashlib
from pathlib import Path
from typing import List

import config
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma



#  PDF loading


def load_pdfs(data_dir: Path = config.DATA_DIR) -> List[Document]:
  
    pages: List[Document] = []
    pdf_paths = sorted(data_dir.glob("*.pdf"))

    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {data_dir}")

    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        raw_pages = loader.load()

        for page in raw_pages:
            # Normalise metadata
            page.metadata["document_name"] = pdf_path.stem
            page.metadata["page_number"] = page.metadata.get("page", 0) + 1
            pages.append(page)

    return pages



#  Chunking with citation metadata


def _make_chunk_id(doc_name: str, page_number: int, content: str) -> str:
    """Return a stable SHA-256-based ID for a chunk."""
    raw = f"{doc_name}:p{page_number}:{content[:120]}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def chunk_documents(
    pages: List[Document],
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> List[Document]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size * 4,        # convert tokens -> approx chars
        chunk_overlap=chunk_overlap * 4,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    raw_chunks = splitter.split_documents(pages)
    chunks: List[Document] = []

    for chunk in raw_chunks:
        doc_name = chunk.metadata.get("document_name", "unknown")
        page_num = chunk.metadata.get("page_number", 0)
        chunk.metadata["chunk_id"] = _make_chunk_id(
            doc_name, page_num, chunk.page_content
        )
        chunks.append(chunk)

    return chunks


# Embedding 

def get_embedding_function(model_name: str = config.EMBEDDING_MODEL, batch_size: int = 32):
    return FastEmbedEmbeddings(model_name=model_name, batch_size=batch_size)


#  Build the Chroma vector index


def build_index(
    chunks: List[Document],
    persist_directory: str = str(config.CHROMA_DIR),
    collection_name: str = config.COLLECTION_NAME,
) -> Chroma:
    from chromadb import PersistentClient
    try:
        client = PersistentClient(path=persist_directory)
        client.delete_collection(collection_name)
    except Exception:
        pass

    embed_fn = get_embedding_function()
    
    # Extract stable IDs from chunk metadata
    ids = [chunk.metadata["chunk_id"] for chunk in chunks]

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embed_fn,
        ids=ids,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    return vectordb

# CLI entry point

if __name__ == "__main__":
    print("=== Day 1 — Document Ingestion Pipeline ===\n")

    print(f"Data directory : {config.DATA_DIR}")
    print(f"Chunk size     : {config.CHUNK_SIZE} tokens")
    print(f"Chunk overlap  : {config.CHUNK_OVERLAP} tokens")
    print(f"Embedding model: {config.EMBEDDING_MODEL}\n")

    print("Loading PDFs...")
    pages = load_pdfs()
    print(f"  Loaded {len(pages)} pages from {config.DATA_DIR}\n")

    print("Chunking documents...")
    chunks = chunk_documents(pages)
    print(f"  Created {len(chunks)} chunks\n")

    print("Building vector index...")
    vectordb = build_index(chunks)
    print(f"Index persisted to {config.CHROMA_DIR}\n")

    print("Done")
