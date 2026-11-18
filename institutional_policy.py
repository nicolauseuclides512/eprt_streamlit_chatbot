"""Kebijakan operasional Pusat Bahasa yang dikonfirmasi pengelola.

Kebijakan ini BUKAN hasil crawling dan BUKAN penetapan nilai/tarif numerik.
Perbarui hanya berdasarkan konfirmasi baru dari pengelola yang berwenang.
"""
CONFIRMED_AT = "2026-11-18"
SOURCE = "Konfirmasi pengelola Pusat Bahasa Telkom University"

POLICY_TEXT = """
Kebijakan operasional yang dikonfirmasi pengelola Pusat Bahasa pada 18 November 2026:
1. Jadwal, biaya, dan ketentuan kelulusan layanan yang sejenis dan untuk kategori
   peserta yang sama berlaku seragam pada seluruh kampus Telkom University.
   Jangan menyatakan ada perbedaan jadwal umum, tarif, atau batas kelulusan hanya
   karena kampusnya Bandung, Purwokerto, Jakarta, Surabaya, atau kampus lain.
2. Untuk kampus Telkom University DI LUAR BANDUNG, penyelenggaraan tes bahasa
   dan kursus bahasa untuk sementara dilakukan DARING/ONLINE.
   Jangan menjanjikan layanan tatap muka di luar Bandung pada kondisi sekarang.
3. Ketentuan mekanisme Bandung tetap mengikuti pengumuman layanan resmi;
   pernyataan ini tidak menetapkan bahwa semua layanan Bandung selalu luring.
4. Program khusus (contoh EPrT mahasiswa baru versus EPrT reguler) tetap harus
   dibedakan berdasarkan jenis program/kategori peserta. Keseragaman antarkampus
   TIDAK berarti semua program gratis atau semua kategori peserta mempunyai
   biaya dan aturan yang sama.
5. Konfirmasi ini tidak memuat angka harga, jadwal tanggal/jam terbaru, nilai
   kelulusan numerik, atau tautan pendaftaran. Jangan mengarang angka tersebut.
   Untuk angka dan slot nyata, gunakan informasi resmi yang berlaku; jika
   belum terverifikasi, jawab belum dapat dipastikan.
6. Sumber kebijakan ini adalah konfirmasi langsung pengelola, bukan situs web;
   bedakan dari kutipan tautan website ketika menjawab.
""".strip()


def policy_related(question: str) -> bool:
    """Izinkan jawaban kebijakan umum walau tidak ada potongan web relevan."""
    q = question.casefold()
    campus_tokens = (
        "kampus", "bandung", "purwokerto", "jakarta", "surabaya", "pjj",
        "luar bandung", "seluruh tel-u", "semua tel-u", "telkom university",
    )
    policy_tokens = (
        "jadwal", "biaya", "harga", "tarif", "kelulusan", "passing grade",
        "ambang", "minimum", "batas lulus", "lulus", "online", "daring",
        "offline", "luring", "kursus", "tes",
    )
    return any(t in q for t in campus_tokens) and any(t in q for t in policy_tokens)
