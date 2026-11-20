"""Offline checks for source attribution and unsupported factual claims."""
import unittest

from answer_guard import make_evidence, validate_answer


class GroundingTests(unittest.TestCase):
    def setUp(self):
        self.evidence = make_evidence([{
            "title": "Panduan resmi EPrT",
            "url": "https://lac.telkomuniversity.ac.id/layanan/tes-bahasa/eprt/",
            "text": "Tes EPrT menguji listening, grammar, reading.",
            "checked_at": "2026-11-20T02:00:00Z",
        }])

    def test_supported_claim_with_citation(self):
        self.assertTrue(validate_answer(
            "EPrT meliputi listening, grammar, dan reading [S1].", self.evidence
        ).ok)

    def test_policy_citation_allowed(self):
        self.assertTrue(validate_answer(
            "Biaya seragam di seluruh kampus Tel-U [P1].", self.evidence
        ).ok)

    def test_no_citation_rejected(self):
        self.assertFalse(validate_answer("Biaya EPrT Rp900.000.", self.evidence).ok)

    def test_fabricated_id_rejected(self):
        self.assertFalse(validate_answer("Harga tetap [S99].", self.evidence).ok)

    def test_fabricated_number_rejected(self):
        result = validate_answer("Biaya Rp900.000 [S1].", self.evidence)
        self.assertFalse(result.ok)
        self.assertIn("Angka", result.message)

    def test_fabricated_link_rejected(self):
        self.assertFalse(validate_answer(
            "Daftar melalui https://unknown.example/register [S1].",
            self.evidence
        ).ok)

    def test_policy_not_claimed_as_website_citation(self):
        result = validate_answer("Semua kampus di luar Bandung online [P1].", self.evidence)
        self.assertEqual(result.referenced, ["P1"])
        self.assertTrue(result.ok)
