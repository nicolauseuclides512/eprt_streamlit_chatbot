"""Regression checks for Gemini API 400 caused by default v1 endpoint."""
import unittest
from unittest.mock import patch

from gemini_client import GEMINI_API_VERSION, create_gemini_client


class GeminiClientTests(unittest.TestCase):
    def test_api_version_is_explicitly_v1beta(self):
        self.assertEqual(GEMINI_API_VERSION, "v1beta")
        with patch("gemini_client.genai.Client") as factory:
            client = create_gemini_client("test-key-placeholder")
        self.assertIs(client, factory.return_value)
        factory.assert_called_once()
        arguments = factory.call_args.kwargs
        self.assertEqual(arguments["api_key"], "test-key-placeholder")
        self.assertEqual(arguments["http_options"].api_version, "v1beta")

    def test_app_uses_shared_client_for_all_gemini_requests(self):
        from pathlib import Path
        source = (Path(__file__).resolve().parent.parent / "app.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("client = create_gemini_client(api_key)", source)
        self.assertNotIn("client = genai.Client(api_key=api_key)", source)


if __name__ == "__main__":
    unittest.main()
