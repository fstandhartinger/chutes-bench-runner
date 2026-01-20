#!/usr/bin/env python3
"""Lookup Artificial Analysis benchmarks for a Chutes model."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

# Ensure backend is on PYTHONPATH when running from repo root
backend_path = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(backend_path))

from app.core.config import get_settings
from app.services.artificial_analysis_service import ArtificialAnalysisService, ChutesModelSummary
from app.services.chutes_client import ChutesClient


async def resolve_chutes_model(identifier: str) -> dict[str, Any] | None:
    settings = get_settings()
    client = ChutesClient(api_key=settings.chutes_api_key)
    models = await client.list_models()
    lowered = identifier.lower()
    for model in models:
        for key in ("slug", "name", "chute_id"):
            value = model.get(key)
            if isinstance(value, str) and value.lower() == lowered:
                return model
    return None


async def main() -> None:
    parser = argparse.ArgumentParser(description="Artificial Analysis bench lookup")
    parser.add_argument("--model", help="Chutes model slug/name/chute_id")
    parser.add_argument("--aa-slug", help="Artificial Analysis slug (skip mapping)")
    parser.add_argument("--include-raw", action="store_true")
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()

    service = ArtificialAnalysisService()

    if args.aa_slug:
        scores = await service.fetch_model_scores(args.aa_slug, include_raw=args.include_raw)
        output = {
            "match": {"slug": args.aa_slug, "method": "direct"},
            "artificial_analysis": scores.__dict__ if scores else None,
        }
        print(json.dumps(output, indent=2))
        return

    if not args.model:
        raise SystemExit("--model or --aa-slug is required")

    model = await resolve_chutes_model(args.model)
    if not model:
        raise SystemExit(f"Chutes model not found: {args.model}")

    summary = ChutesModelSummary(
        slug=model.get("slug") or model.get("name") or args.model,
        name=model.get("name") or model.get("slug") or args.model,
        tagline=model.get("tagline"),
        user=model.get("user"),
        chute_id=model.get("chute_id"),
    )

    match, scores = await service.get_benchmarks_for_model(
        summary,
        include_raw=args.include_raw,
        llm_fallback=not args.no_llm,
    )

    output = {
        "model": model,
        "match": match.__dict__,
        "artificial_analysis": scores.__dict__ if scores else None,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
