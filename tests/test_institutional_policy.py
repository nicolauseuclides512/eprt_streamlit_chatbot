"""Tes kebijakan operasional yang dikonfirmasi pengelola, tanpa API key."""
import unittest
from institutional_policy import POLICY_TEXT, policy_related, CONFIRMED_AT, SOURCE


class InstitutionalPolicyTests(unittest.TestCase):
    def test_shared_rules(self):
        self.assertIn("seragam", POLICY_TEXT)
        self.assertIn("jadwal", POLICY_TEXT)
        self.assertIn("biaya", POLICY_TEXT)
        self.assertIn("kelulusan", POLICY_TEXT)
        self.assertIn("seluruh kampus Telkom University", POLICY_TEXT)

    def test_non_bandung_online_both_service_types(self):
        self.assertIn("DI LUAR BANDUNG", POLICY_TEXT)
        self.assertIn("tes bahasa", POLICY_TEXT)
        self.assertIn("kursus bahasa", POLICY_TEXT)
        self.assertIn("DARING/ONLINE", POLICY_TEXT)

    def test_operator_not_public_web_or_numeric_policy(self):
        self.assertIn("konfirmasi langsung pengelola", POLICY_TEXT)
        self.assertIn("jangan mengarang", POLICY_TEXT.casefold())
        self.assertEqual(CONFIRMED_AT, "2026-11-18")
        self.assertIn("Pusat Bahasa", SOURCE)

    def test_policy_questions_without_website_chunks(self):
        for question in (
            "Apakah biaya EPrT di Purwokerto berbeda dengan Bandung?",
            "Apakah tes dan kursus luar Bandung masih online?",
            "Apakah ketentuan kelulusan di seluruh Tel-U sama?",
        ):
            self.assertTrue(policy_related(question), question)

    def test_unrelated_question_requires_web_evidence(self):
        self.assertFalse(policy_related("Bagaimana mengunduh sertifikat?"))
        self.assertFalse(policy_related("Apa alamat email LaC?"))


if __name__ == "__main__":
    unittest.main()
