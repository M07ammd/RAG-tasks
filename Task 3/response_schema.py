"""JSON Schema and validation for grounded clinical responses."""

from typing import Any, Dict

from jsonschema import validate


RESPONSE_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object", "additionalProperties": False,
    "required": ["status", "recommendation", "evidence", "citations", "confidence"],
    "properties": {
        "status": {"enum": ["grounded", "refused"]},
        "recommendation": {"type": ["string", "null"]},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "citations": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["document", "page"],
            "properties": {
                "document": {"type": "string", "minLength": 1},
                "page": {"type": "integer", "minimum": 1},
                "chunk_id": {"type": "string", "minLength": 1},
            },
        }},
        "confidence": {"enum": ["high", "medium", "low", "insufficient"]},
    },
    "allOf": [
        {"if": {"properties": {"status": {"const": "grounded"}}},
         "then": {"properties": {"recommendation": {"type": "string", "minLength": 1},
                                    "evidence": {"minItems": 1}, "citations": {"minItems": 1}},
                  "required": ["recommendation", "evidence", "citations"]}},
        {"if": {"properties": {"status": {"const": "refused"}}},
         "then": {"properties": {"recommendation": {"type": "null"}, "evidence": {"maxItems": 0},
                                    "citations": {"maxItems": 0}, "confidence": {"const": "insufficient"}}}},
        {"if": {"properties": {"confidence": {"const": "high"}}},
         "then": {"properties": {"evidence": {"minItems": 1}, "citations": {"minItems": 1}}}},
    ],
}


def refusal_response() -> Dict[str, Any]:
    return {"status": "refused", "recommendation": None, "evidence": [],
            "citations": [], "confidence": "insufficient"}


def validate_response_schema(response: Dict[str, Any]) -> Dict[str, Any]:
    validate(instance=response, schema=RESPONSE_SCHEMA)
    return response