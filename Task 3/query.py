from pathlib import Path
from typing import List, Tuple

import config
from ingest import get_embedding_function
from langchain_chroma import Chroma
from langchain_core.documents import Document


def load_index(
    persist_directory: str = str(config.CHROMA_DIR),
    collection_name: str = config.COLLECTION_NAME,
    embedding_model: str = config.EMBEDDING_MODEL,
) -> Chroma:
    """
    Load an existing Chroma vector index from disk.
    """
    embed_fn = get_embedding_function(model_name=embedding_model)

    vectordb = Chroma(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_function=embed_fn,
    )
    return vectordb


def retrieve(
    vectordb: Chroma,
    query: str,
    k: int = 3,
) -> List[Tuple[Document, float]]:
    return vectordb.similarity_search_with_relevance_scores(query, k=k)


if __name__ == "__main__":
    print("=== Vector Index Query ===\n")
    print(f"Loading index from: {config.CHROMA_DIR}")
    print(f"Collection name   : {config.COLLECTION_NAME}")
    print(f"Embedding model   : {config.EMBEDDING_MODEL}\n")

    vectordb = load_index()

    sample_query = "What is the target blood pressure for a patient with cardiovascular disease?"
    print(f"Sample query: {sample_query}\n")

    results = retrieve(vectordb, sample_query, k=3)
    for i, (doc, score) in enumerate(results, 1):
        doc_name = doc.metadata.get("document_name", "unknown")
        page_num = doc.metadata.get("page_number", "unknown")
        print(f"[{i}] score={score:.3f} | doc={doc_name} | page={page_num}")
        print(f"    {doc.page_content[:150].strip()}...\n")
