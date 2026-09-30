import unittest

from jev_policy_gate.fake import DeterministicFakeJevClient


class TestFakeDeterminism(unittest.TestCase):
    def test_same_input_same_output(self):
        client = DeterministicFakeJevClient()
        questions = {
            "go": {"type": "noul", "instructions": "go?", "criteria": {"true": {}, "false": {}}},
            "risk": {"type": "score", "instructions": "risk?", "criteria": [{"what": "a"}]},
        }
        state = {"flight": {"delay_minutes": 45}}

        r1 = client.decide(state, questions)
        r2 = client.decide(state, questions)

        self.assertEqual(r1["go"].value, r2["go"].value)
        self.assertEqual(r1["go"].confidence, r2["go"].confidence)
        self.assertEqual(r1["risk"].value, r2["risk"].value)
        self.assertEqual(r1["risk"].raw["label"], r2["risk"].raw["label"])

    def test_different_input_different_output(self):
        client = DeterministicFakeJevClient()
        questions = {"go": {"type": "noul", "instructions": "go?", "criteria": {}}}

        r1 = client.decide({"a": 1}, questions)
        r2 = client.decide({"a": 2}, questions)

        self.assertNotEqual(r1["go"].value, r2["go"].value)

    def test_continuous_confidence_spread_not_discrete(self):
        client = DeterministicFakeJevClient()
        confidences = set()
        for i in range(50):
            questions = {"go": {"type": "noul", "instructions": f"go-{i}?", "criteria": {}}}
            result = client.decide({"i": i}, questions)
            confidences.add(round(result["go"].confidence, 6))

        # A continuous spread should produce (close to) 50 distinct values,
        # not collapse onto a handful of fixed points.
        self.assertGreater(len(confidences), 40)
        self.assertGreaterEqual(min(confidences), 0.05)
        self.assertLessEqual(max(confidences), 0.95)

    def test_response_shape_matches_real_api(self):
        client = DeterministicFakeJevClient()
        questions = {
            "go": {"type": "noul", "instructions": "go?", "criteria": {}},
            "pick": {
                "type": "choice",
                "instructions": "pick?",
                "criteria": {"left": {"what": "l"}, "right": {"what": "r"}},
            },
            "risk": {"type": "score", "instructions": "risk?", "criteria": [{"what": "a"}]},
        }
        result = client.decide({}, questions)

        self.assertIsInstance(result["go"].value, float)
        self.assertTrue(0.0 <= result["go"].value <= 1.0)

        self.assertIn(result["pick"].value, ("left", "right"))

        self.assertIsInstance(result["risk"].value, float)
        self.assertTrue(0.0 <= result["risk"].value <= 1.0)
        self.assertIn("legend", result["risk"].raw)
        self.assertIn("label", result["risk"].raw)


if __name__ == "__main__":
    unittest.main()
