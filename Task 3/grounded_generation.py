"""Backward-compatible public API for the modular grounded pipeline."""

from typing import Any, Dict, Sequence, Tuple

from langchain_core.documents import Document

from citation_validator import validate_citations
from grounding_prompt import GROUNDING_SYSTEM_PROMPT, build_grounded_prompt, format_context
from pipeline import generate_grounded_response
from response_schema import RESPONSE_SCHEMA, refusal_response, validate_response_schema


def validate_response(response: Dict[str, Any], retrieved_chunks: Sequence[Tuple[Document, float]]) -> Dict[str, Any]:
    validate_response_schema(response)
    validate_citations(response, retrieved_chunks)
    return response