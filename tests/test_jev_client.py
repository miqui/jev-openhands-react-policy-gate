import json
import unittest
import urllib.error
from io import BytesIO
from unittest import mock

from jev_policy_gate.jev_client import (
    JevDecisionClient,
    JevDecisionError,
    _legend_label,
)


class _FakeHTTPResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _mock_urlopen(response_obj):
    body = json.dumps(response_obj).encode("utf-8")
    return _FakeHTTPResponse(body)


class TestRequestShape(unittest.TestCase):
    def setUp(self):
        self.client = JevDecisionClient(api_key="test-key")

    def _capture_request(self, answers):
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["url"] = req.full_url
            captured["method"] = req.get_method()
            captured["headers"] = dict(req.header_items())
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return _mock_urlopen({"answers": answers})

        return captured, fake_urlopen

    def test_noul_question_shape_and_answer(self):
        questions = {
            "go": {
                "type": "noul",
                "instructions": "Should we go?",
                "criteria": {
                    "true": {"what": "safe"},
                    "false": {"what": "unsafe"},
                },
            }
        }
        captured, fake_urlopen = self._capture_request({"go": {"value": 0.8, "confidence": 0.9}})
        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            result = self.client.decide({"s": 1}, questions)

        self.assertEqual(captured["method"], "POST")
        self.assertEqual(captured["url"], "https://openrouter.ai/api/alpha/decisions")
        self.assertEqual(captured["body"]["model"], "typesafe/jev-1.13")
        self.assertEqual(captured["body"]["state"], {"s": 1})
        self.assertIn("instructions", captured["body"]["questions"]["go"])
        self.assertNotIn("text", captured["body"]["questions"]["go"])
        self.assertIsInstance(captured["body"]["questions"]["go"]["criteria"], dict)
        self.assertEqual(captured["headers"]["Authorization"], "Bearer test-key")

        self.assertIn("go", result)
        self.assertAlmostEqual(result["go"].value, 0.8)
        self.assertAlmostEqual(result["go"].confidence, 0.9)

    def test_choice_question_criteria_is_record_not_array(self):
        questions = {
            "pick": {
                "type": "choice",
                "instructions": "Pick one option.",
                "criteria": {
                    "left": {"what": "go left"},
                    "right": {"what": "go right"},
                },
            }
        }
        captured, fake_urlopen = self._capture_request({"pick": {"value": "left", "confidence": 0.5}})
        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            result = self.client.decide({}, questions)

        criteria = captured["body"]["questions"]["pick"]["criteria"]
        self.assertIsInstance(criteria, dict)
        self.assertNotIsInstance(criteria, list)
        self.assertNotIn("options", captured["body"]["questions"]["pick"])
        self.assertIn("instructions", captured["body"]["questions"]["pick"])

        self.assertEqual(result["pick"].value, "left")

    def test_score_question_criteria_is_array_of_levels(self):
        questions = {
            "risk": {
                "type": "score",
                "instructions": "How risky is this `state.foo`?",
                "criteria": [
                    {"what": "low", "signals": ["a"]},
                    {"what": "medium"},
                    {"what": "high"},
                ],
            }
        }
        legend = {
            "0": {"what": "very low"},
            "1": {"what": "low"},
            "2": {"what": "moderate"},
            "3": {"what": "high"},
            "4": {"what": "very high"},
        }
        captured, fake_urlopen = self._capture_request(
            {"risk": {"value": 0.42, "confidence": 0.7, "legend": legend}}
        )
        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            result = self.client.decide({}, questions)

        criteria = captured["body"]["questions"]["risk"]["criteria"]
        self.assertIsInstance(criteria, list)
        self.assertIn("instructions", captured["body"]["questions"]["risk"])

        ans = result["risk"]
        self.assertAlmostEqual(ans.value, 0.42)
        # bucket = min(4, floor(0.42*5)) = min(4, floor(2.1)) = 2 -> "moderate"
        self.assertEqual(ans.raw["label"], "moderate")


class TestLegendLabelMatching(unittest.TestCase):
    def test_standard_bucketing(self):
        legend = {
            "0": {"what": "very low"},
            "1": {"what": "low"},
            "2": {"what": "moderate"},
            "3": {"what": "high"},
            "4": {"what": "very high"},
        }
        self.assertEqual(_legend_label(0.0, legend), "very low")
        self.assertEqual(_legend_label(0.19, legend), "very low")
        self.assertEqual(_legend_label(0.2, legend), "low")
        self.assertEqual(_legend_label(0.99, legend), "very high")
        self.assertEqual(_legend_label(1.0, legend), "very high")

    def test_never_uses_arithmetic_shortcut(self):
        # Regression: label must come from legend[bucket]['what'], not a formula.
        legend = {
            "0": {"what": "zzz-low"},
            "1": {"what": "zzz-mid"},
            "2": {"what": "zzz-high"},
            "3": {"what": "zzz-higher"},
            "4": {"what": "zzz-max"},
        }
        self.assertEqual(_legend_label(0.5, legend), "zzz-high")

    def test_score_vs_legend_count_mismatch(self):
        # Legend only has 3 levels (0,1,2) though score is 0-1 with bucket 0-4.
        legend = {
            "0": {"what": "low"},
            "1": {"what": "mid"},
            "2": {"what": "high"},
        }
        # bucket for 0.9 -> floor(4.5)=4, not in legend -> fallback to closest <= 4 -> "2" -> high
        self.assertEqual(_legend_label(0.9, legend), "high")
        # bucket for 0.05 -> 0 -> present directly
        self.assertEqual(_legend_label(0.05, legend), "low")

    def test_empty_legend_returns_none(self):
        self.assertIsNone(_legend_label(0.5, {}))
        self.assertIsNone(_legend_label(0.5, None))


class TestErrorHandling(unittest.TestCase):
    def setUp(self):
        self.client = JevDecisionClient(api_key="test-key")

    def test_http_400_raises_jev_decision_error_with_path(self):
        error_body = json.dumps(
            {"error": "Bad Request", "path": ["questions", "go", "instructions"]}
        ).encode("utf-8")

        def fake_urlopen(req, timeout=None):
            raise urllib.error.HTTPError(
                url=req.full_url, code=400, msg="Bad Request", hdrs=None, fp=BytesIO(error_body)
            )

        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            with self.assertRaises(JevDecisionError):
                self.client.decide({}, {"go": {"type": "noul", "instructions": "x", "criteria": {}}})

    def test_malformed_json_raises(self):
        def fake_urlopen(req, timeout=None):
            return _FakeHTTPResponse(b"not json{{{")

        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            with self.assertRaises(JevDecisionError):
                self.client.decide({}, {"go": {"type": "noul", "instructions": "x", "criteria": {}}})

    def test_missing_answers_key_raises(self):
        def fake_urlopen(req, timeout=None):
            return _mock_urlopen({"not_answers": {}})

        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            with self.assertRaises(JevDecisionError):
                self.client.decide({}, {"go": {"type": "noul", "instructions": "x", "criteria": {}}})

    def test_single_missing_answer_field_fails_only_that_question(self):
        questions = {
            "go": {"type": "noul", "instructions": "x", "criteria": {}},
            "risk": {"type": "score", "instructions": "y", "criteria": []},
        }

        def fake_urlopen(req, timeout=None):
            # 'risk' answer entirely absent from the response.
            return _mock_urlopen({"answers": {"go": {"value": 0.6, "confidence": 0.8}}})

        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            result = self.client.decide({}, questions)

        self.assertIn("go", result)
        self.assertNotIn("risk", result)

    def test_malformed_individual_answer_is_skipped(self):
        questions = {
            "go": {"type": "noul", "instructions": "x", "criteria": {}},
        }

        def fake_urlopen(req, timeout=None):
            return _mock_urlopen({"answers": {"go": {"value": "not-a-number"}}})

        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            result = self.client.decide({}, questions)

        self.assertNotIn("go", result)

    def test_missing_api_key_raises(self):
        with self.assertRaises(JevDecisionError):
            JevDecisionClient(api_key="")


if __name__ == "__main__":
    unittest.main()
