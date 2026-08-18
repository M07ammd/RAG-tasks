"""Single entry point for the complete Task 2 + Task 3 system."""

import argparse
import json

from pipeline import generate_grounded_response
from query import load_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the grounded clinical RAG system")
    parser.add_argument("question", nargs="?", default="What is the target blood pressure for a patient with cardiovascular disease?")
    parser.add_argument("--mode", choices=["valid", "refusal", "invalid_schema", "invalid_citation"], default="valid")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--provider", choices=["gemini", "ollama", "simulation"], default=None)
    args = parser.parse_args()

    if args.provider:
        import config
        config.LLM_PROVIDER = args.provider

    vector_index = load_index()
    options = {"simulation_mode": args.mode}
    if args.threshold is not None:
        options["confidence_threshold"] = args.threshold
    result = generate_grounded_response(args.question, vector_index, **options)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()