"""End-to-end grounded clinical RAG pipeline."""

import json
import os
from typing import Any, Dict

from jsonschema import ValidationError

import config
from citation_validator import validate_citations
from grounding_prompt import build_grounded_prompt
from llm_service import generate_gemini_response, generate_local_response, simulate_response
from query import retrieve
from response_schema import refusal_response, validate_response_schema


def generate_grounded_response(query: str, vectordb: Any, k: int = 3,
                               confidence_threshold: float = config.GROUNDING_THRESHOLD,
                               simulation_mode: str = "valid") -> Dict[str, Any]:
    """Retrieve, gate, generate, and validate one grounded response."""
    try:
        retrieved_chunks = retrieve(vectordb, query, k=k)
    except (OSError, ValueError, RuntimeError) as error:
        raise RuntimeError(f"Task 2 retrieval failed: {error}") from error

    if not retrieved_chunks or retrieved_chunks[0][1] < confidence_threshold:
        return refusal_response()

    prompt = build_grounded_prompt(query, retrieved_chunks)
    try:
        if simulation_mode != "valid":
            response = simulate_response(retrieved_chunks, simulation_mode)
        elif config.LLM_PROVIDER == "gemini" and config.GEMINI_API_KEY:
            response = generate_gemini_response(prompt)
        elif config.LLM_PROVIDER == "ollama":
            response = generate_local_response(prompt)
        else:
            response = simulate_response(retrieved_chunks, simulation_mode)
        validate_response_schema(response)
        validate_citations(response, retrieved_chunks)
        return response
    except (ValidationError, ValueError, json.JSONDecodeError, ImportError,
            OSError, RuntimeError, ConnectionError) as error:
        raise ValueError(f"Generated response was rejected: {error}") from error