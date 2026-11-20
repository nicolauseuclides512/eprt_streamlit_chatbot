"""Konstruksi Gemini API client untuk RAG dan generasi jawaban.

google-genai 3.0+ memakai v1 sebagai default. Untuk konfigurasi
system_instruction chatbot ini, endpoint v1beta harus dipilih eksplisit
agar tidak terjadi 400 "Developer instruction is not enabled for api version v1".
"""
from google import genai
from google.genai import types

GEMINI_API_VERSION = "v1beta"


def create_gemini_client(api_key: str) -> genai.Client:
    """Gunakan versi API yang konsisten untuk embedding dan generasi."""
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(api_version=GEMINI_API_VERSION),
    )
