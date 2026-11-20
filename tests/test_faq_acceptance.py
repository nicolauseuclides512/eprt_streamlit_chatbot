"""Deterministic FAQ routing acceptance checks, not Gemini E2E tests."""
import json
import unittest
from pathlib import Path

from faq_rules import answer_faq, load_faq

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "faq_scenarios.json"


class FaqAcceptanceTests(unittest.TestCase):
    def test_all_offline_question_scenarios(self):
        fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(fixture["cases"]), 50)
        rules = load_faq()
        for item in fixture["cases"]:
            with self.subTest(question=item["question"]):
                found = answer_faq(item["question"], rules)
                expected = item["expected_rule"]
                if expected is None:
                    self.assertIsNone(found)
                else:
                    self.assertIsNotNone(found)
                    self.assertIn(expected, found["rule_ids"])
                    self.assertIn("Sumber:", found["answer"])

    def test_policy_is_auditable_and_has_kg_ready_fields(self):
        rules = load_faq()["rules"]
        for rule in rules:
            self.assertEqual(rule["status"], "active")
            for key in ("id", "subject", "predicate", "object",
                        "scope", "valid_from", "valid_until"):
                self.assertIn(key, rule)
            self.assertTrue(rule["answer"])
