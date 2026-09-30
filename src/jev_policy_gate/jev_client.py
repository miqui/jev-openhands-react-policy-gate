"""JevDecisionClient: thin urllib-based client for the Jev decision API.

Jev is a DECISION model only. This module hits:

    POST https://openrouter.ai/api/alpha/decisions
    Authorization: Bearer $OPENROUTER_API_KEY
    {"model": "typesafe/jev-1.13", "state": {...}, "questions": {...}}

Per-question failures (missing/malformed answer) are tolerated: that answer is
simply omitted from the returned dict, rather than raising for the whole batch.
Only transport-level failures (HTTP error, totally unparsable body, wrong
top-level shape) raise JevDecisionError.
"""
from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from dataclasses import dataclass

DEFAULT_MODEL = "typesafe/jev-1.13"
DEFAULT_BASE_URL = "https://openrouter.ai/api/alpha/decisions"


class JevDecisionError(Exception):
    """Raised on HTTP error, malformed top-level response, or transport failure."""


@dataclass(frozen=True)
class JevAnswer:
    name: str
    value: float | str
    confidence: float | None
    raw: dict


def _clamp01(x: float) -> float:
    if math.isnan(x):
        return 0.0
    return max(0.0, min(1.0, float(x)))


def _legend_label(score: float, legend: dict) -> str | None:
    """Bucket a normalized 0-1 score into a legend level and return its label.

    bucket = min(4, floor(score * 5)); label-match only against legend[str(bucket)],
    NEVER arithmetic like score*4+k. Handles legends with fewer than 5 entries
    (score-vs-legend-count mismatch) by falling back to the closest available
    bucket key, and returns None if the legend is empty/unusable.
    """
    if not isinstance(legend, dict) or not legend:
        return None
    bucket = min(4, math.floor(_clamp01(score) * 5))
    key = str(bucket)
    if key not in legend:
        # Score-vs-legend-count mismatch: legend may have fewer levels than the
        # standard 0..4. Find the closest available integer key <= bucket, else
        # the closest overall.
        candidates = []
        for k in legend:
            try:
                candidates.append(int(k))
            except (TypeError, ValueError):
                continue
        if not candidates:
            return None
        candidates.sort()
        lower = [c for c in candidates if c <= bucket]
        chosen = max(lower) if lower else min(candidates)
        key = str(chosen)
        if key not in legend:
            return None
    level = legend[key]
    if isinstance(level, dict):
        return level.get("what")
    if isinstance(level, str):
        return level
    return None


class JevDecisionClient:
    """Real HTTP client for the Jev decision API, stdlib urllib only."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
    ) -> None:
        if not api_key:
            raise JevDecisionError("api_key is required (source from env var only)")
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._timeout = timeout

    def decide(self, state: dict, questions: dict[str, dict]) -> dict[str, JevAnswer]:
        payload = {"model": self._model, "state": state, "questions": questions}
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._base_url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                raw_body = resp.read()
        except urllib.error.HTTPError as exc:
            err_body = exc.read()
            try:
                parsed = json.loads(err_body.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                parsed = None
            path = None
            if isinstance(parsed, dict):
                path = parsed.get("path") or (
                    parsed.get("error", {}).get("path") if isinstance(parsed.get("error"), dict) else None
                )
            raise JevDecisionError(
                f"Jev API HTTP {exc.code}: {parsed if parsed is not None else err_body!r}"
                + (f" (path={path})" if path else "")
            ) from exc
        except urllib.error.URLError as exc:
            raise JevDecisionError(f"Jev API request failed: {exc}") from exc

        try:
            parsed_body = json.loads(raw_body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise JevDecisionError(f"Jev API returned malformed JSON: {exc}") from exc

        if not isinstance(parsed_body, dict):
            raise JevDecisionError("Jev API response is not a JSON object")

        answers_raw = parsed_body.get("answers")
        if not isinstance(answers_raw, dict):
            raise JevDecisionError("Jev API response missing 'answers' object")

        results: dict[str, JevAnswer] = {}
        for name, question in questions.items():
            if name not in answers_raw:
                # Per-question failure tolerance: skip, don't raise.
                continue
            ans = answers_raw[name]
            if not isinstance(ans, dict):
                continue
            qtype = question.get("type")
            try:
                results[name] = self._parse_answer(name, qtype, ans)
            except JevDecisionError:
                # Malformed individual answer: tolerate, skip only this one.
                continue
        return results

    @staticmethod
    def _parse_answer(name: str, qtype: str, ans: dict) -> JevAnswer:
        confidence = ans.get("confidence")
        if confidence is not None:
            try:
                confidence = _clamp01(float(confidence))
            except (TypeError, ValueError):
                confidence = None

        if qtype == "noul":
            if "value" not in ans:
                raise JevDecisionError(f"answer '{name}' missing 'value'")
            try:
                value = _clamp01(float(ans["value"]))
            except (TypeError, ValueError) as exc:
                raise JevDecisionError(f"answer '{name}' non-numeric noul value") from exc
            return JevAnswer(name=name, value=value, confidence=confidence, raw=ans)

        if qtype == "choice":
            if "value" not in ans or not isinstance(ans["value"], str):
                raise JevDecisionError(f"answer '{name}' missing/invalid choice 'value'")
            return JevAnswer(name=name, value=ans["value"], confidence=confidence, raw=ans)

        if qtype == "score":
            if "value" not in ans:
                raise JevDecisionError(f"answer '{name}' missing 'value'")
            try:
                score = _clamp01(float(ans["value"]))
            except (TypeError, ValueError) as exc:
                raise JevDecisionError(f"answer '{name}' non-numeric score value") from exc
            legend = ans.get("legend", {})
            label = _legend_label(score, legend)
            raw = dict(ans)
            raw["label"] = label
            return JevAnswer(name=name, value=score, confidence=confidence, raw=raw)

        raise JevDecisionError(f"answer '{name}' has unknown question type {qtype!r}")
