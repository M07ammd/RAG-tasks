"""Simulation and optional live LLM calls."""

import json
import re
from urllib.parse import quote
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from typing import Any, Dict, Sequence, Tuple

import config
from langchain_core.documents import Document
from response_schema import refusal_response


def simulate_response(retrieved_chunks: Sequence[Tuple[Document, float]], mode: str = "valid") -> Dict[str, Any]:
    if mode == "refusal":
        return refusal_response()
    if mode == "invalid_schema":
        return {"status": "grounded", "recommendation": "Unsupported medical recommendation",
                "evidence": [], "citations": [], "confidence": "high"}
    if mode == "invalid_citation":
        return {"status": "grounded", "recommendation": "Unsupported citation test", "evidence": ["test"],
                "citations": [{"document": "Fake Medical Guideline", "page": 999}], "confidence": "high"}
    if not retrieved_chunks:
        return refusal_response()
    document, score = retrieved_chunks[0]
    metadata = document.metadata
    return {"status": "grounded", "recommendation": document.page_content[:240].strip(),
            "evidence": [document.page_content[:500].strip()],
            "citations": [{"document": metadata.get("document_name", "unknown"),
                           "page": int(metadata.get("page_number", 1)),
                           "chunk_id": metadata.get("chunk_id", "unknown")}],
            "confidence": "high" if score >= 0.7 else "medium"}


def generate_local_response(prompt: str) -> Dict[str, Any]:
    """Call a local Ollama server and parse its JSON response."""
    payload = json.dumps({
        "model": config.LOCAL_LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
    }).encode("utf-8")
    request = Request(
        f"{config.LOCAL_LLM_URL.rstrip('/')}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=180) as response:
            body = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama returned HTTP {error.code}: {detail}") from error
    except URLError as error:
        raise RuntimeError(
            f"Cannot connect to Ollama at {config.LOCAL_LLM_URL}. "
            "Start Ollama and pull the configured model."
        ) from error

    content = body.get("message", {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Ollama returned no message content")
    return json.loads(content)


def generate_gemini_response(prompt: str) -> Dict[str, Any]:
    """Call Gemini using its REST API and parse the JSON-only response."""
    if not config.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{quote(config.GEMINI_MODEL, safe='')}:generateContent"
        f"?key={quote(config.GEMINI_API_KEY, safe='')}"
    )
    payload = json.dumps({
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
    }).encode("utf-8")
    request = Request(endpoint, data=payload,
                      headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=180) as response:
            body = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini returned HTTP {error.code}: {detail}") from error
    except URLError as error:
        raise RuntimeError("Cannot connect to the Gemini API") from error

    try:
        content = body["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError(f"Gemini returned an unexpected response: {body}") from error
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Gemini returned no text content")
    response = json.loads(content)
    return _normalize_gemini_response(response)


def _normalize_gemini_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """Map Gemini's common citation alias to the project schema field."""
    citations = response.get("citations")
    if not isinstance(citations, list):
        return response
    normalized_citations = []
    for citation in citations:
        if isinstance(citation, str):
            citation_match = re.match(
                r"^(?P<document>.+),\s*Page\s+(?P<page>\d+),\s*Chunk ID\s+(?P<chunk_id>\S+)$",
                citation.strip(),
                flags=re.IGNORECASE,
            )
            if citation_match:
                citation = citation_match.groupdict()
                citation["page"] = int(citation["page"])
        if isinstance(citation, dict):
            citation = dict(citation)
            if "source" in citation and "document" not in citation:
                citation["document"] = citation.pop("source")
            if "source_id" in citation and "chunk_id" not in citation:
                citation["chunk_id"] = citation.pop("source_id")
            if isinstance(citation.get("page"), str) and citation["page"].isdigit():
                citation["page"] = int(citation["page"])
        normalized_citations.append(citation)
    normalized_response = dict(response)
    normalized_response["citations"] = normalized_citations
    return normalized_response