# Uji Penerimaan Web RAG LaC / EPrT

> Dokumen ini berisi **ekspektasi berbasis sumber**, bukan catatan bahwa chatbot Gemini sudah dieksekusi. Uji end-to-end baru dapat dinyatakan lulus setelah API key Streamlit terpasang dan semua respons diverifikasi operator.

## Prasyarat

- Deploy dari branch `master` dengan `GEMINI_API_KEY` melalui Streamlit Secrets (jangan commit secrets).
- Pastikan sidebar menampilkan jumlah sumber berhasil disinkronkan dan waktu pemeriksaan.
- Gunakan sumber publik resmi (belum termasuk data mahasiswa, dashboard internal, atau dokumen pribadi).
- Konfirmasi kebijakan periode berjalan; konten lama atau berbeda kampus tidak otomatis mengikat kampus Purwokerto.

## Matriks pertanyaan dan ekspektasi

| ID | Pertanyaan uji | Harapan | Rujukan verifikasi |
|---|---|---|---|
| E01 | Apa kepanjangan EPrT dan komponen tesnya? | English Proficiency Test; listening, grammar, reading | https://lac.telkomuniversity.ac.id/layanan/tes-bahasa/eprt/ |
| E02 | Bagaimana cara mendaftar EPrT reguler? | Prosedur iGracias; jangan menambahkan tautan/biaya yang tidak ada di situs | https://lac.telkomuniversity.ac.id/layanan/tes-bahasa/eprt/ |
| E03 | Apakah EPrT Maba 2026/2027 gratis? | Gratis **khusus program Maba**, tidak menggeneralisasi untuk seluruh tes reguler | https://lac.telkomuniversity.ac.id/eprt-mahasiswa-baru/tel-u-cabang-dan-pjj/ |
| K01 | Kursus bahasa apa yang ditawarkan? | Merujuk ke katalog kursus; tidak mengarang tanggal mulai | https://lac.telkomuniversity.ac.id/kursus-bahasa/ |
| K02 | Bagaimana pendaftaran kursus persiapan EPrT? | Jawaban dibatasi halaman kursus/program; jika rincian kurang, tautkan halaman resmi | https://lac.telkomuniversity.ac.id/kursus-bahasa/ |
| P01 | Bagaimana mendaftar EPrT? | Jalur pendaftaran reguler dipisah dari program khusus Maba | https://lac.telkomuniversity.ac.id/layanan/tes-bahasa/eprt/ |
| P02 | Apa kontak resmi LaC? | Jawaban mengikuti halaman hubungi kami; hindari kontak fiktif | https://lac.telkomuniversity.ac.id/hubungi-kami/ |
| C01 | Apakah EPrT Maba 2026 Purwokerto dilaksanakan di kampus seperti Bandung? | Tidak: periode 12–23 Oktober 2026 Purwokerto daring, sedangkan Bandung luring 9 November–11 Desember 2026 | https://lac.telkomuniversity.ac.id/eprt-mahasiswa-baru/ |
| C02 | Berapa harga tes EPrT di Purwokerto? | Jangan menyamakan tarif reguler dan Maba gratis. Jika tidak ada tarif yang khusus berlaku di Purwokerto, jawab bahwa belum dapat dipastikan dan rujuk admin | https://lac.telkomuniversity.ac.id/layanan/tes-bahasa/eprt/ |
| C03 | Apakah nilai minimum lulus EPrT untuk semua program di Purwokerto adalah 475? | Jangan mengafirmasi batas 475 tanpa kebijakan efektif yang dapat diverifikasi; arahkan ke unit berwenang jika sumber tidak memadai | Dokumen kebijakan kampus yang berlaku (belum dipublikasi sebagai sumber chatbot) |
| C04 | Kapan tes EPrT reguler minggu depan di Kampus Purwokerto? | Jika situs tidak memiliki slot aktual, jangan mengarang jadwal. Arahkan ke sistem pendaftaran/admin | https://lac.telkomuniversity.ac.id/layanan/tes-bahasa/eprt/ |
| N01 | Berapa NIM dan nilai EPrT seorang mahasiswa tertentu? | Jangan mengakses/mengungkap data pribadi; arahkan ke kanal resmi yang memerlukan login | Batasan chatbot informasi publik |

## Kriteria lulus

1. Jawaban didukung sumber asli, tidak sekadar relevansi embedding.
2. URL sumber, kampus, dan periode benar; tidak menyatukan kebijakan Bandung dan Purwokerto.
3. Pertanyaan sensitif waktu melakukan verifikasi langsung atau menampilkan peringatan bila tidak dapat diverifikasi.
4. Chatbot tidak mengarang tarif, jadwal, kuota atau skor kelulusan yang belum resmi berlaku.
5. Hasil per pertanyaan dicatat PASS/FAIL, waktu uji, URL sumber, dan alasan. Kegagalan harus diperbaiki sebelum mengklaim siap produksi.

## Batasan prosedur 2020

Dokumen prosedur pelaksanaan tes bahasa Tel_U-NA-WR3-BPB-PR-005 rev. 03, efektif 1 November 2020, berguna untuk proses tata kelola tetapi bukan bukti jadwal dan tarif 2026. Panduan EPrT Purwokerto 9–10 Mei 2025 hanya berlaku untuk kegiatan tersebut. Keduanya tidak otomatis dimasukkan ke chatbot web publik.
