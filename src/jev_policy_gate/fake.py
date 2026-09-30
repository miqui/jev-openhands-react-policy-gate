"""DeterministicFakeJevClient: hash-derived, keyless, network-free fake for tests.

Produces the same response *shape* as the real Jev API (answers with
value/confidence, score answers with a legend), but derives every number
deterministically from a stable hash of (tool_name/state/questions), with a
CONTINUOUS confidence spread (not a few discrete buckets).
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from .jev_client import JevAnswer, _legend_label


def _stable_hash(*parts: Any) -> int:
    payload = json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:16], 16)


def _unit_float(*parts: Any) -> float:
    """Deterministic float in [0, 1), continuously distributed across inputs."""
    h = _stable_hash(*parts)
    return (h % 10_000_000) / 10_000_000.0


_DEFAULT_LEGEND = {
    "0": {"what": "very low"},
    "1": {"what": "low"},
    "2": {"what": "moderate"},
    "3": {"what": "high"},
    "4": {"what": "very high"},
}


class DeterministicFakeJevClient:
    """Keyless, network-free stand-in for JevDecisionClient.

    Same public surface: decide(state, questions) -> dict[str, JevAnswer].
    All values are pure functions of the input, so identical calls produce
    identical results (determinism), while different inputs spread
    continuously over the confidence range (e.g. ~0.05..0.95) rather than
    collapsing onto a few fixed values.
    """

    def __init__(self, legend: dict | None = None) -> None:
        self._legend = legend if legend is not None else _DEFAULT_LEGEND

    def decide(self, state: dict, questions: dict[str, dict]) -> dict[str, JevAnswer]:
        results: dict[str, JevAnswer] = {}
        for name, question in questions.items():
            qtype = question.get("type")
            base = _unit_float(name, question.get("instructions"), state)
            # Continuous confidence spread across a wide band, never a few
            # discrete points: 0.05 .. 0.95.
            confidence = 0.05 + 0.90 * _unit_float("confidence", name, state)

            if qtype == "noul":
                value = base
                raw = {"value": value, "confidence": confidence}
                results[name] = JevAnswer(name=name, value=value, confidence=confidence, raw=raw)

            elif qtype == "choice":
                criteria = question.get("criteria") or {}
                options = list(criteria.keys()) if isinstance(criteria, dict) else []
                if not options:
                    continue
                idx = _stable_hash("choice-pick", name, state) % len(options)
                chosen = options[idx]
                raw = {"value": chosen, "confidence": confidence}
                results[name] = JevAnswer(name=name, value=chosen, confidence=confidence, raw=raw)

            elif qtype == "score":
                score = base
                legend = self._legend
                label = _legend_label(score, legend)
                raw = {"value": score, "confidence": confidence, "legend": legend, "label": label}
                results[name] = JevAnswer(name=name, value=score, confidence=confidence, raw=raw)

            # Unknown types are simply skipped (per-question tolerance).
        return results
