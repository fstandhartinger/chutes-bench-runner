"""Artificial Analysis lookup and mapping utilities."""
from __future__ import annotations

from dataclasses import dataclass
import json
import re
import time
from typing import Any, Optional

import httpx

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.chutes_client import get_chutes_client

logger = get_logger(__name__)
settings = get_settings()


@dataclass(frozen=True)
class ChutesModelSummary:
    slug: str
    name: str
    tagline: Optional[str] = None
    user: Optional[str] = None
    chute_id: Optional[str] = None
    readme: Optional[str] = None


@dataclass(frozen=True)
class MatchResult:
    slug: Optional[str]
    method: str
    confidence: Optional[float]
    candidates: list[str]
    llm_used: bool = False
    notes: Optional[str] = None


@dataclass(frozen=True)
class ArtificialAnalysisScores:
    slug: str
    name: Optional[str]
    short_name: Optional[str]
    model_url: Optional[str]
    hosts_url: Optional[str]
    scores: list[dict[str, Any]]
    raw: Optional[dict[str, Any]] = None


SCORE_LABELS: dict[str, str] = {
    "aime25": "AIME 2025 (Competition Math)",
    "livecodebench": "LiveCodeBench v6 (Coding)",
    "terminalbench_hard": "Terminal-Bench Hard (Agentic Coding)",
    "mmlu_pro": "MMLU-Pro (General Knowledge)",
    "tau2": "τ²-Bench Telecom (Tool Use)",
    "gpqa": "GPQA Diamond (Scientific Reasoning)",
    "hle": "Humanity's Last Exam",
    "scicode": "SciCode (Coding)",
    "ifbench": "IFBench (Instruction Following)",
    "gdpval": "GDPval-AA",
    "lcr": "AA-LCR (Long Context Reasoning)",
    "omniscience": "AA-Omniscience (Overall)",
    "coding_index": "Coding Index",
    "intelligence_index": "Intelligence Index",
    "math_index": "Math Index",
}


class ArtificialAnalysisService:
    """Fetch and map Artificial Analysis scores for Chutes models."""

    def __init__(self) -> None:
        self.base_url = settings.artificial_analysis_base_url
        self.sitemap_url = settings.artificial_analysis_sitemap_url
        self.timeout = settings.artificial_analysis_timeout_seconds
        self._slug_cache: list[str] = []
        self._slug_cache_at: float = 0.0

    async def list_model_slugs(self) -> list[str]:
        ttl = settings.artificial_analysis_cache_ttl_seconds
        now = time.monotonic()
        if self._slug_cache and (now - self._slug_cache_at) < ttl:
            return self._slug_cache

        async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout, connect=10.0)) as client:
            response = await client.get(self.sitemap_url, headers={"User-Agent": "chutes-bench-runner"})
            response.raise_for_status()
            xml = response.text

        urls = re.findall(r"<loc>(.*?)</loc>", xml)
        slugs: list[str] = []
        for url in urls:
            if not url.startswith(f"{self.base_url}/models/"):
                continue
            slug = url.split("/models/", 1)[-1]
            if not slug or "/" in slug:
                continue
            if slug in {"recommend", "multilingual", "caching"}:
                continue
            slugs.append(slug)

        self._slug_cache = sorted(set(slugs))
        self._slug_cache_at = now
        return self._slug_cache

    async def fetch_model_scores(
        self,
        slug: str,
        include_raw: bool = False,
    ) -> Optional[ArtificialAnalysisScores]:
        url = f"{self.base_url}/models/{slug}"
        async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout, connect=10.0)) as client:
            response = await client.get(url, headers={"User-Agent": "chutes-bench-runner"})
            if response.status_code == 404:
                return None
            response.raise_for_status()
            html = response.text

        decoded = self._decode_next_f(html)
        payload = self._extract_model_payload(decoded, slug)
        if not payload:
            return None

        scores = []
        for key, label in SCORE_LABELS.items():
            value = payload.get(key)
            if isinstance(value, (int, float)):
                scores.append(
                    {
                        "key": key,
                        "label": label,
                        "value": float(value),
                        "format": "percent" if 0 <= value <= 1.5 else "raw",
                    }
                )

        model_url = payload.get("model_url")
        if isinstance(model_url, str) and model_url.startswith("/"):
            model_url = f"{self.base_url}{model_url}"
        hosts_url = payload.get("hosts_url")
        if isinstance(hosts_url, str) and hosts_url.startswith("/"):
            hosts_url = f"{self.base_url}{hosts_url}"

        return ArtificialAnalysisScores(
            slug=payload.get("slug", slug),
            name=payload.get("name"),
            short_name=payload.get("short_name"),
            model_url=model_url,
            hosts_url=hosts_url,
            scores=scores,
            raw=payload if include_raw else None,
        )

    async def map_chutes_model(
        self,
        model: ChutesModelSummary,
        llm_fallback: bool = True,
    ) -> MatchResult:
        slugs = await self.list_model_slugs()
        candidates = self._generate_candidates(model)

        for candidate in candidates:
            if candidate in slugs:
                return MatchResult(candidate, "exact", 1.0, candidates, llm_used=False)

        for candidate in candidates:
            prefix_matches = [slug for slug in slugs if slug.startswith(candidate)]
            if len(prefix_matches) == 1:
                return MatchResult(prefix_matches[0], "prefix", 0.9, candidates, llm_used=False)

        best_slug = None
        best_score = 0.0
        scored: list[tuple[str, float]] = []
        for slug in slugs:
            score = max(self._similarity(candidate, slug) for candidate in candidates)
            scored.append((slug, score))
            if score > best_score:
                best_score = score
                best_slug = slug

        scored.sort(key=lambda item: item[1], reverse=True)
        top_candidates = [slug for slug, _ in scored[:25]]

        if best_slug and best_score >= 0.78:
            return MatchResult(best_slug, "fuzzy", best_score, top_candidates, llm_used=False)

        if llm_fallback and settings.artificial_analysis_llm_fallback:
            llm_match = await self._llm_match(model, top_candidates)
            if llm_match:
                return llm_match

        return MatchResult(None, "none", None, top_candidates, llm_used=bool(settings.artificial_analysis_llm_fallback))

    async def build_model_summary(self, model: ChutesModelSummary) -> ChutesModelSummary:
        if not model.chute_id:
            return model
        if not settings.chutes_api_key:
            return model

        url = f"{settings.chutes_models_api_url}/chutes/{model.chute_id}"
        headers = {"Authorization": f"Bearer {settings.chutes_api_key}"}
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout, connect=10.0)) as client:
                resp = await client.get(url, headers=headers)
                if resp.status_code >= 400:
                    return model
                data = resp.json()
        except Exception as exc:
            logger.debug("Failed to fetch chute details", error=str(exc))
            return model

        readme = None
        for key in ("readme", "README", "description", "about", "long_description"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                readme = value.strip()
                break
        if readme:
            return ChutesModelSummary(
                slug=model.slug,
                name=model.name,
                tagline=model.tagline,
                user=model.user,
                chute_id=model.chute_id,
                readme=readme,
            )
        return model

    async def get_benchmarks_for_model(
        self,
        model: ChutesModelSummary,
        include_raw: bool = False,
        llm_fallback: bool = True,
    ) -> tuple[MatchResult, Optional[ArtificialAnalysisScores]]:
        enriched = await self.build_model_summary(model)
        match = await self.map_chutes_model(enriched, llm_fallback=llm_fallback)
        if not match.slug:
            return match, None
        scores = await self.fetch_model_scores(match.slug, include_raw=include_raw)
        if not scores:
            return match, None
        return match, scores

    def _decode_next_f(self, html: str) -> str:
        fragments = re.findall(r"self.__next_f.push\(\[1,\"(.*?)\"\]\)\</script>", html)
        decoded_parts: list[str] = []
        for fragment in fragments:
            try:
                decoded_parts.append(json.loads('"' + fragment + '"'))
                continue
            except Exception:
                pass
            try:
                decoded_parts.append(
                    json.loads('"' + fragment.replace('\\"', '"').replace('\\\\', '\\') + '"')
                )
            except Exception:
                continue
        return "".join(decoded_parts)

    def _extract_model_payload(self, decoded: str, slug: str) -> Optional[dict[str, Any]]:
        """
        Extract the JSON payload for a given model slug from the decoded Next.js stream.

        The AA /models/<slug> pages currently contain multiple JSON blobs that repeat the same
        `"slug":"<slug>"` substring (for example: a small "model card" object and a larger
        "performance metrics" object). The older implementation grabbed the *first* match which
        often lacks benchmark fields, resulting in an empty scores list.
        """

        needle = f'"slug":"{slug}"'
        indices = [m.start() for m in re.finditer(re.escape(needle), decoded)]
        if not indices:
            return None

        best_payload: Optional[dict[str, Any]] = None
        best_score_key_hits = -1
        best_size = -1

        for idx in indices:
            payload = self._extract_json_object_around(decoded, idx)
            if not payload:
                continue
            score_key_hits = len(set(payload.keys()).intersection(SCORE_LABELS.keys()))
            size = len(payload.keys())
            candidate = (score_key_hits, size)
            best = (best_score_key_hits, best_size)
            if candidate > best:
                best_payload = payload
                best_score_key_hits = score_key_hits
                best_size = size

        return best_payload

    def _extract_json_object_around(self, decoded: str, idx: int) -> Optional[dict[str, Any]]:
        """Extract the JSON object enclosing `idx` (which must point inside the object text)."""
        start: Optional[int] = None
        depth = 0
        for i in range(idx, -1, -1):
            ch = decoded[i]
            if ch == "}":
                depth += 1
            elif ch == "{":
                if depth == 0:
                    start = i
                    break
                depth -= 1
        if start is None:
            return None

        level = 0
        end: Optional[int] = None
        for i in range(start, len(decoded)):
            ch = decoded[i]
            if ch == "{":
                level += 1
            elif ch == "}":
                level -= 1
                if level == 0:
                    end = i
                    break
        if end is None:
            return None

        try:
            parsed = json.loads(decoded[start : end + 1])
        except Exception as exc:
            logger.debug("Failed to parse AA payload", error=str(exc))
            return None
        return parsed if isinstance(parsed, dict) else None

    def _normalize(self, value: str) -> str:
        text = value.lower().strip()
        text = text.replace("/", "-")
        text = text.replace("_", "-")
        text = text.replace(".", "-")
        text = re.sub(r"[^a-z0-9\-]+", "-", text)
        text = re.sub(r"-+", "-", text).strip("-")
        return text

    def _strip_suffixes(self, value: str) -> str:
        suffixes = [
            "-tee",
            "-instruct",
            "-instruction",
            "-chat",
            "-reasoning",
            "-non-reasoning",
            "-thinking",
            "-fp8",
            "-int8",
            "-int4",
            "-awq",
            "-gptq",
            "-gguf",
            "-q4",
            "-q5",
            "-q6",
            "-q8",
            "-4bit",
            "-8bit",
            "-preview",
            "-beta",
        ]
        for suffix in suffixes:
            if value.endswith(suffix):
                return value[: -len(suffix)]
        return value

    def _generate_candidates(self, model: ChutesModelSummary) -> list[str]:
        raw_values = [model.slug, model.name, model.tagline or "", model.user or ""]
        candidates: list[str] = []
        for raw in raw_values:
            if not raw:
                continue
            base = raw.split("/")[-1]
            normalized = self._normalize(base)
            if normalized:
                candidates.append(normalized)
                candidates.append(self._strip_suffixes(normalized))
        # dedupe and filter
        cleaned = []
        for candidate in candidates:
            candidate = candidate.strip("-")
            if not candidate or candidate in cleaned:
                continue
            cleaned.append(candidate)
        return cleaned

    def _similarity(self, a: str, b: str) -> float:
        if a == b:
            return 1.0
        if a and b and (a.startswith(b) or b.startswith(a)):
            return 0.9
        from difflib import SequenceMatcher

        return SequenceMatcher(None, a, b).ratio()

    async def _llm_match(self, model: ChutesModelSummary, candidates: list[str]) -> Optional[MatchResult]:
        if not candidates:
            return None
        system_prompt = (
            "You are a strict JSON-only assistant that maps Chutes model identifiers "
            "to Artificial Analysis slugs. Return ONLY JSON with keys: slug, confidence, reason. "
            "The slug must be one of the candidates or null."
        )
        prompt = (
            "Pick the best Artificial Analysis model slug for this Chutes model.\n\n"
            f"Chutes model slug: {model.slug}\n"
            f"Name: {model.name}\n"
            f"Tagline: {model.tagline or 'n/a'}\n"
            f"User: {model.user or 'n/a'}\n"
            f"Readme: {(model.readme or 'n/a')[:1200]}\n\n"
            "Candidates:\n"
            + "\n".join(f"- {slug}" for slug in candidates)
            + "\n\nReturn JSON only."
        )

        client = get_chutes_client()
        mapper_model = settings.artificial_analysis_mapper_model
        try:
            text, _ = await client.get_completion_text(
                mapper_model,
                prompt,
                system_prompt=system_prompt,
                temperature=0.0,
                max_tokens=350,
                response_attempts=2,
            )
        except Exception as exc:
            logger.warning("LLM mapping failed", error=str(exc))
            return None

        if not text:
            return None
        match = self._extract_json(text)
        if not match:
            return None
        slug = match.get("slug") if isinstance(match.get("slug"), str) else None
        if slug not in candidates:
            return None
        confidence = match.get("confidence")
        if not isinstance(confidence, (int, float)):
            confidence = None
        return MatchResult(slug, "llm", float(confidence) if confidence is not None else None, candidates, llm_used=True)

    def _extract_json(self, text: str) -> Optional[dict[str, Any]]:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
