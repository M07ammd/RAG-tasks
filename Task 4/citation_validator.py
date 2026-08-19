"""Independent validation of generated citations against retrieved chunks."""

from typing import Any, Dict, Sequence, Tuple

from langchain_core.documents import Document


def validate_citations(response: Dict[str, Any], retrieved_chunks: Sequence[Tuple[Document, float]]) -> None:
    if response["status"] == "refused":
        return
    sources = {(str(document.metadata.get("document_name", "unknown")),
                int(document.metadata.get("page_number", 0)),
                str(document.metadata.get("chunk_id", "")))
               for document, _ in retrieved_chunks}
    for citation in response["citations"]:
        matching_pages = [source for source in sources
                          if source[0] == citation["document"] and source[1] == citation["page"]]
        if not matching_pages:
            raise ValueError(f"Citation is not present in retrieved context: {citation}")
        if citation.get("chunk_id") and (citation["document"], citation["page"], citation["chunk_id"]) not in sources:
            raise ValueError(f"Citation chunk is not present in retrieved context: {citation}")