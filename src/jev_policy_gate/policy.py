"""JevPolicyGate: enforced policy layer sitting in front of tool execution.

Sends ONE batched Jev request per evaluate() call: a `noul` go/no-go question
and a `score` risk question. Routes to ALLOW / ESCALATE / DENY using
constructor-provided thresholds only (never magic numbers in the logic).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .jev_client import JevDecisionError

GO_QUESTION = "go_no_go"
RISK_QUESTION = "risk_score"


class GateAction(Enum):
    ALLOW = "ALLOW"
    DENY = "DENY"
    ESCALATE = "ESCALATE"


@dataclass(frozen=True)
class GateDecision:
    action: GateAction
    confidence: float
    reason: str
    jev_raw: dict


class JevPolicyGate:
    def __init__(
        self,
        client: Any,
        allow_threshold: float = 0.70,
        deny_risk_threshold: float = 0.70,
        go_field: str = "value",
    ) -> None:
        """
        client: an object exposing decide(state, questions) -> dict[str, JevAnswer]
            (JevDecisionClient or DeterministicFakeJevClient).
        allow_threshold: minimum confidence required, when Jev says "go", to ALLOW
            outright rather than ESCALATE.
        deny_risk_threshold: minimum normalized risk score, when present, that
            forces DENY regardless of the go/no-go answer.
        """
        self._client = client
        self._allow_threshold = allow_threshold
        self._deny_risk_threshold = deny_risk_threshold
        self._go_field = go_field
        self._audit_log: list[dict] = []

    def _build_questions(self, tool_name: str, action_args: dict) -> dict[str, dict]:
        return {
            GO_QUESTION: {
                "type": "noul",
                "instructions": (
                    f"Should the agent be allowed to execute tool `{tool_name}` with the given "
                    "arguments right now? Consider `action_args` and `state` for risk signals."
                ),
                "criteria": {
                    "true": {
                        "what": "The action is safe/expected and should proceed.",
                    },
                    "false": {
                        "what": "The action is unsafe, destructive, or out of policy.",
                    },
                },
            },
            RISK_QUESTION: {
                "type": "score",
                "instructions": (
                    f"How risky is executing tool `{tool_name}` with the given arguments, "
                    "on a normalized scale?"
                ),
                "criteria": [
                    {"what": "negligible risk, fully reversible"},
                    {"what": "low risk"},
                    {"what": "moderate risk"},
                    {"what": "high risk"},
                    {"what": "severe/irreversible risk"},
                ],
            },
        }

    def evaluate(self, tool_name: str, action_args: dict, state: dict) -> GateDecision:
        questions = self._build_questions(tool_name, action_args)
        full_state = dict(state)
        full_state["tool_name"] = tool_name
        full_state["action_args"] = action_args

        try:
            answers = self._client.decide(full_state, questions)
        except JevDecisionError as exc:
            decision = GateDecision(
                action=GateAction.ESCALATE,
                confidence=0.0,
                reason=f"Jev decision request failed: {exc}",
                jev_raw={},
            )
            self._record(tool_name, action_args, decision)
            return decision

        go_answer = answers.get(GO_QUESTION)
        risk_answer = answers.get(RISK_QUESTION)

        jev_raw = {
            GO_QUESTION: go_answer.raw if go_answer else None,
            RISK_QUESTION: risk_answer.raw if risk_answer else None,
        }

        if go_answer is None:
            decision = GateDecision(
                action=GateAction.ESCALATE,
                confidence=0.0,
                reason="Jev did not return a go/no-go answer for this action.",
                jev_raw=jev_raw,
            )
            self._record(tool_name, action_args, decision)
            return decision

        go = bool(go_answer.value >= 0.5) if isinstance(go_answer.value, (int, float)) else bool(go_answer.value)
        confidence = go_answer.confidence if go_answer.confidence is not None else 0.0
        risk_score = risk_answer.value if risk_answer is not None else None

        if isinstance(risk_score, (int, float)) and risk_score >= self._deny_risk_threshold:
            decision = GateDecision(
                action=GateAction.DENY,
                confidence=confidence,
                reason=f"Risk score {risk_score:.2f} >= deny_risk_threshold {self._deny_risk_threshold:.2f}.",
                jev_raw=jev_raw,
            )
        elif not go:
            decision = GateDecision(
                action=GateAction.DENY,
                confidence=confidence,
                reason="Jev go/no-go answer was no-go.",
                jev_raw=jev_raw,
            )
        elif confidence >= self._allow_threshold:
            decision = GateDecision(
                action=GateAction.ALLOW,
                confidence=confidence,
                reason=f"Go with confidence {confidence:.2f} >= allow_threshold {self._allow_threshold:.2f}.",
                jev_raw=jev_raw,
            )
        else:
            decision = GateDecision(
                action=GateAction.ESCALATE,
                confidence=confidence,
                reason=f"Go but confidence {confidence:.2f} < allow_threshold {self._allow_threshold:.2f}.",
                jev_raw=jev_raw,
            )

        self._record(tool_name, action_args, decision)
        return decision

    def _record(self, tool_name: str, action_args: dict, decision: GateDecision) -> None:
        self._audit_log.append(
            {
                "tool_name": tool_name,
                "action_args": action_args,
                "action": decision.action.value,
                "confidence": decision.confidence,
                "reason": decision.reason,
                "jev_raw": decision.jev_raw,
            }
        )

    def audit_log(self) -> list[dict]:
        return list(self._audit_log)
