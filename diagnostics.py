"""Pesan diagnostik aman untuk kegagalan chatbot Gemini/FAISS.

Jangan pernah menampilkan str(exception) mentah di aplikasi: pesan API
dapat mengandung prompt atau data pengguna. Detail lengkap hanya di log server.
"""


def friendly_error(exc: Exception, stage: str) -> str:
    """Ubah error API/jaringan menjadi petunjuk singkat tanpa membocorkan rahasia."""
    errors = []
    seen = set()
    current = exc
    while current is not None and id(current) not in seen:
        errors.append(current)
        seen.add(id(current))
        current = current.__cause__ or current.__context__

    codes = set()
    for error in errors:
        for attribute in ("code", "status_code"):
            value = getattr(error, attribute, None)
            try:
                if value is not None:
                    codes.add(int(value))
            except (TypeError, ValueError):
                pass

    # Hanya digunakan untuk mencocokkan tipe error; jangan diperlihatkan mentah.
    raw = " ".join(str(error) for error in errors).upper()
    prefix = f"**Gagal pada tahap {stage}.** "

    if "API_KEY_INVALID" in raw or "API KEY NOT VALID" in raw or "API KEY EXPIRED" in raw:
        return prefix + "API key Gemini tidak valid atau sudah tidak berlaku. Periksa `GEMINI_API_KEY` dalam `.streamlit/secrets.toml` lalu mulai ulang Streamlit."
    if 429 in codes or "RESOURCE_EXHAUSTED" in raw or "QUOTA" in raw or "RATE LIMIT" in raw:
        return prefix + "Batas penggunaan (kuota/rate limit) Gemini tercapai. Periksa kuota proyek di Google AI Studio, lalu coba lagi setelah batasnya tersedia."
    if 403 in codes or "PERMISSION_DENIED" in raw:
        return prefix + "Akses ke model Gemini ditolak (403). Periksa izin API key, proyek, dan ketersediaan model."
    if 401 in codes or "UNAUTHENTICATED" in raw:
        return prefix + "Autentikasi Gemini gagal (401). Periksa konfigurasi API key."
    if 404 in codes or "NOT_FOUND" in raw and "MODEL" in raw:
        return prefix + "Model Gemini tidak ditemukan atau tidak tersedia untuk API key ini (404). Periksa `GEMINI_MODEL` pada konfigurasi."
    if 400 in codes or "INVALID_ARGUMENT" in raw:
        return prefix + "Permintaan ke Gemini ditolak (400). Periksa kompatibilitas model dan parameter aplikasi; detail tersedia di terminal Streamlit."
    if "EMPTY_GENERATION_RESPONSE" in raw:
        return prefix + "Gemini tidak menghasilkan teks jawaban. Coba ulang atau tinjau respons dan pembatasan model pada log."
    if 503 in codes or "UNAVAILABLE" in raw:
        return prefix + "Layanan Gemini sedang tidak tersedia (503). Coba kembali setelah beberapa saat."
    if 500 in codes or 502 in codes or 504 in codes:
        return prefix + "Layanan model mengalami gangguan sementara (5xx). Coba kembali setelah beberapa saat."
    if any(token in raw for token in ("TIMEOUT", "TIMED OUT", "CONNECTIONERROR", "CONNECTERROR", "NAME RESOLUTION")):
        return prefix + "Koneksi ke layanan gagal atau melebihi batas waktu. Periksa internet, firewall/proxy, kemudian coba lagi."

    return prefix + "Terjadi kesalahan yang belum teridentifikasi. Lihat traceback di terminal tempat Streamlit dijalankan (jangan bagikan API key atau informasi rahasia)."
