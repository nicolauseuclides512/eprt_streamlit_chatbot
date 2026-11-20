"""Tes diagnostik yang tidak membutuhkan akses Gemini."""
import unittest
from diagnostics import friendly_error


class DummyApiError(Exception):
    def __init__(self, code, details):
        super().__init__(details)
        self.code = code


class FriendlyErrorTests(unittest.TestCase):
    def test_invalid_api_key(self):
        response = friendly_error(DummyApiError(400, "API_KEY_INVALID - secret-token"), "pembuatan jawaban Gemini")
        self.assertIn("API key", response)
        self.assertNotIn("secret-token", response)

    def test_exhausted_quota(self):
        response = friendly_error(DummyApiError(429, "RESOURCE_EXHAUSTED"), "pencarian referensi")
        self.assertIn("kuota", response)
        self.assertIn("pencarian referensi", response)

    def test_model_not_found(self):
        self.assertIn("Model Gemini", friendly_error(DummyApiError(404, "NOT_FOUND"), "pembuatan jawaban Gemini"))

    def test_denied_access(self):
        self.assertIn("Akses", friendly_error(DummyApiError(403, "PERMISSION_DENIED"), "pembuatan jawaban Gemini"))

    def test_invalid_argument(self):
        self.assertIn("400", friendly_error(DummyApiError(400, "INVALID_ARGUMENT"), "pembuatan jawaban Gemini"))

    def test_empty_response(self):
        self.assertIn("tidak menghasilkan teks", friendly_error(RuntimeError("EMPTY_GENERATION_RESPONSE"), "pembuatan jawaban Gemini"))

    def test_network_timeout(self):
        self.assertIn("Koneksi", friendly_error(TimeoutError("timed out while connecting"), "pencarian referensi"))

    def test_unknown_error_never_leaks_data(self):
        response = friendly_error(RuntimeError("sensitive-prompt-content"), "pembuatan jawaban Gemini")
        self.assertIn("terminal", response)
        self.assertNotIn("sensitive-prompt-content", response)

    def test_wrapped_error_preserves_original_code(self):
        try:
            try:
                raise DummyApiError(429, "quota")
            except DummyApiError as cause:
                raise RuntimeError("Gemini gagal") from cause
        except RuntimeError as err:
            self.assertIn("kuota", friendly_error(err, "pembuatan jawaban Gemini"))


if __name__ == "__main__":
    unittest.main()
