"""Grounding prompt and retrieved-context formatting."""

from typing import Sequence, Tuple

from langchain_core.documents import Document


GROUNDING_SYSTEM_PROMPT = """You are a citation-bound clinical assistant that answers only from the supplied retrieved guideline evidence. You are not an unrestricted medical advisor.

Context boundary:
- Use ONLY the supplied retrieved context.
- Do not use external medical knowledge or prior model knowledge.
- Do not infer unsupported clinical facts or invent recommendations.
- Every clinical claim must be traceable to retrieved evidence.
- If evidence is insufficient, irrelevant, out of scope, or does not support the requested claim, refuse.

Output rules:
- Return JSON only, with no Markdown or text outside the JSON object.
- The JSON must match the response schema: status, recommendation, evidence, citations, and confidence.
- A grounded response must include non-empty evidence and citations from the supplied sources.

Refusal rule:
- For insufficient or irrelevant evidence, return status "refused", recommendation null, empty evidence and citations, and confidence "insufficient".
"""


def format_context(retrieved_chunks: Sequence[Tuple[Document, float]]) -> str:
    sections = []
    for index, (document, score) in enumerate(retrieved_chunks, 1):
        metadata = document.metadata
        sections.append(
            f"[Source {index}]\nDocument: {metadata.get('document_name', 'unknown')}\n"
            f"Page: {metadata.get('page_number', 0)}\nChunk ID: {metadata.get('chunk_id', 'unknown')}\n"
            f"Retrieval Score: {score:.4f}\n\n{document.page_content}"
        )
    return "\n\n".join(sections)


def build_grounded_prompt(query: str, retrieved_chunks: Sequence[Tuple[Document, float]]) -> str:
    return f"{GROUNDING_SYSTEM_PROMPT}\n\nRetrieved context:\n{format_context(retrieved_chunks)}\n\nQuestion: {query}"