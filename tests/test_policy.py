import unittest

from jev_policy_gate.jev_client import JevAnswer, JevDecisionError
from jev_policy_gate.policy import GateAction, JevPolicyGate


class _StubClient:
    """Test double implementing the JevDecisionClient protocol precisely."""

    def __init__(self, answers=None, calls_log=None, raise_error=None):
        self._answers = answers or {}
        self._calls_log = calls_log if calls_log is not None else []
        self._raise_error = raise_error

    def decide(self, state, questions):
        self._calls_log.append({"state": state, "questions": questions})
        if self._raise_error:
            raise self._raise_error
        return self._answers


def _go_answer(value, confidence):
    return JevAnswer(name="go_no_go", value=value, confidence=confidence, raw={"value": value, "confidence": confidence})


def _risk_answer(value):
    return JevAnswer(name="risk_score", value=value, confidence=0.9, raw={"value": value})


class TestBatchedRequest(unittest.TestCase):
    def test_evaluate_sends_exactly_one_request(self):
        calls = []
        client = _StubClient(answers={"go_no_go": _go_answer(1.0, 0.9)}, calls_log=calls)
        gate = JevPolicyGate(client)

        gate.evaluate("run_shell", {"cmd": "ls"}, {"workspace": "/tmp"})

        self.assertEqual(len(calls), 1)
        questions = calls[0]["questions"]
        self.assertIn("go_no_go", questions)
        self.assertIn("risk_score", questions)
        self.assertEqual(questions["go_no_go"]["type"], "noul")
        self.assertEqual(questions["risk_score"]["type"], "score")


class TestThresholdRouting(unittest.TestCase):
    def test_allow_when_go_and_high_confidence(self):
        client = _StubClient(answers={"go_no_go": _go_answer(1.0, 0.85)})
        gate = JevPolicyGate(client, allow_threshold=0.70)

        decision = gate.evaluate("read_file", {}, {})

        self.assertEqual(decision.action, GateAction.ALLOW)
        self.assertAlmostEqual(decision.confidence, 0.85)

    def test_escalate_when_go_but_low_confidence(self):
        client = _StubClient(answers={"go_no_go": _go_answer(1.0, 0.40)})
        gate = JevPolicyGate(client, allow_threshold=0.70)

        decision = gate.evaluate("read_file", {}, {})

        self.assertEqual(decision.action, GateAction.ESCALATE)

    def test_deny_when_no_go(self):
        client = _StubClient(answers={"go_no_go": _go_answer(0.0, 0.95)})
        gate = JevPolicyGate(client, allow_threshold=0.70)

        decision = gate.evaluate("rm_rf", {}, {})

        self.assertEqual(decision.action, GateAction.DENY)

    def test_deny_when_risk_score_exceeds_threshold_even_if_go(self):
        client = _StubClient(
            answers={
                "go_no_go": _go_answer(1.0, 0.95),
                "risk_score": _risk_answer(0.9),
            }
        )
        gate = JevPolicyGate(client, allow_threshold=0.70, deny_risk_threshold=0.70)

        decision = gate.evaluate("delete_prod_db", {}, {})

        self.assertEqual(decision.action, GateAction.DENY)

    def test_thresholds_are_constructor_args(self):
        client = _StubClient(answers={"go_no_go": _go_answer(1.0, 0.55)})
        gate_strict = JevPolicyGate(client, allow_threshold=0.90)
        gate_lenient = JevPolicyGate(client, allow_threshold=0.50)

        self.assertEqual(gate_strict.evaluate("x", {}, {}).action, GateAction.ESCALATE)
        self.assertEqual(gate_lenient.evaluate("x", {}, {}).action, GateAction.ALLOW)

    def test_escalate_when_client_raises(self):
        client = _StubClient(raise_error=JevDecisionError("boom"))
        gate = JevPolicyGate(client)

        decision = gate.evaluate("x", {}, {})

        self.assertEqual(decision.action, GateAction.ESCALATE)

    def test_escalate_when_go_answer_missing(self):
        client = _StubClient(answers={})
        gate = JevPolicyGate(client)

        decision = gate.evaluate("x", {}, {})

        self.assertEqual(decision.action, GateAction.ESCALATE)


class TestAuditLog(unittest.TestCase):
    def test_every_decision_is_logged(self):
        client = _StubClient(answers={"go_no_go": _go_answer(1.0, 0.9)})
        gate = JevPolicyGate(client)

        gate.evaluate("a", {"x": 1}, {})
        gate.evaluate("b", {"y": 2}, {})

        log = gate.audit_log()
        self.assertEqual(len(log), 2)
        self.assertEqual(log[0]["tool_name"], "a")
        self.assertEqual(log[1]["tool_name"], "b")
        for entry in log:
            self.assertIn("action", entry)
            self.assertIn("confidence", entry)
            self.assertIn("reason", entry)
            self.assertIn("jev_raw", entry)
            # JSONL-serializable check
            import json

            json.dumps(entry)


if __name__ == "__main__":
    unittest.main()
