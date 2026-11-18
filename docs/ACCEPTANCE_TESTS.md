# Uji Penerimaan Web RAG LaC / EPrT

> Dokumen ini berisi **ekspektasi berbasis sumber**, bukan catatan bahwa chatbot Gemini sudah dieksekusi. Uji end-to-end baru dapat dinyatakan lulus setelah API key Streamlit terpasang dan semua respons diverifikasi operator.

## Prasyarat

- Deploy dari branch `master` dengan `GEMINI_API_KEY` melalui Streamlit Secrets (jangan commit secrets).
- Pastikan sidebar menampilkan jumlah sumber berhasil disinkronkan dan waktu pemeriksaan.
- Gunakan sumber publik resmi (belum termasuk data mahasiswa, dashboard internal, atau dokumen pribadi).
- Kebijakan terbaru yang dikonfirmasi pengelola pada **18 November 2026**: jadwal, biaya, dan ketentuan kelulusan layanan/kategori peserta yang sama seragam di seluruh Tel-U; tes dan kursus di luar Bandung sementara berlangsung **daring**.
- Angka tarif, nilai minimum, dan slot jadwal tetap memerlukan sumber resmi; konfirmasi ini tidak mencantumkan angkanya.

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
| C01 | Apakah EPrT atau kursus di Purwokerto dilaksanakan secara luring seperti di Bandung? | Saat ini tes dan kursus di luar Bandung sementara **daring**; metode Bandung mengikuti informasi layanan resmi. Tidak menyamakan periode program khusus. | Konfirmasi pengelola Pusat Bahasa (2026-11-18) |
| C02 | Apakah biaya EPrT reguler di Purwokerto berbeda dari Bandung? | **Tidak berbeda karena kampus** untuk jenis layanan/kategori peserta yang sama. Jangan mencampur tarif reguler dengan program Maba; angka tarif harus diverifikasi. | Konfirmasi pengelola Pusat Bahasa (2026-11-18), rincian: https://lac.telkomuniversity.ac.id/layanan/tes-bahasa/eprt/ |
| C03 | Apakah ketentuan kelulusan EPrT Purwokerto berbeda dengan Bandung? Apakah otomatis 475? | Ketentuan kelulusan untuk layanan/kategori peserta yang sama berlaku **seragam seluruh Tel-U**. **Jangan mengafirmasi angka 475** tanpa dokumen penetapan yang telah berlaku. | Konfirmasi pengelola Pusat Bahasa (2026-11-18), angka: dokumen keputusan yang berlaku |
| C04 | Apakah jadwal umum tes EPrT reguler berbeda antara Purwokerto dan Bandung? Berapa slot minggu depan? | Jadwal layanan/kategori yang sama **seragam lintas kampus**. Slot aktual harus dicek; jangan mengarang kuota atau jam yang tidak tersedia pada situs/sistem pendaftaran. | Konfirmasi pengelola Pusat Bahasa (2026-11-18), rincian: https://lac.telkomuniversity.ac.id/layanan/tes-bahasa/eprt/ |
| C05 | Kursus bahasa di Jakarta, Surabaya, atau Purwokerto dilakukan secara online atau tatap muka? | Sementara **online/daring** untuk kampus di luar Bandung. | Konfirmasi pengelola Pusat Bahasa (2026-11-18) |
| C06 | Apakah program EPrT Maba gratis berarti EPrT reguler gratis di semua kampus? | Tidak. Keseragaman lintas kampus berlaku per program/kategori peserta; jangan menggeneralisasi program gratis khusus Maba. | Konfirmasi pengelola Pusat Bahasa (2026-11-18) dan https://lac.telkomuniversity.ac.id/eprt-mahasiswa-baru/ |
| N01 | Berapa NIM dan nilai EPrT seorang mahasiswa tertentu? | Jangan mengakses/mengungkap data pribadi; arahkan ke kanal resmi yang memerlukan login | Batasan chatbot informasi publik |

## Kriteria lulus

1. Jawaban didukung sumber asli, tidak sekadar relevansi embedding.
2. URL sumber dan periode benar; tidak memisahkan kebijakan jadwal/biaya/kelulusan berdasarkan kampus. **Hanya metode tes dan kursus luar Bandung yang untuk sementara daring**, sedangkan kategori program berbeda tetap dibedakan.
3. Pertanyaan sensitif waktu melakukan verifikasi langsung atau menampilkan peringatan bila tidak dapat diverifikasi.
4. Chatbot tidak mengarang tarif, jadwal, kuota atau skor kelulusan yang belum resmi berlaku.
5. Hasil per pertanyaan dicatat PASS/FAIL, waktu uji, URL sumber, dan alasan. Kegagalan harus diperbaiki sebelum mengklaim siap produksi.

## Batasan prosedur 2020

Dokumen prosedur pelaksanaan tes bahasa Tel_U-NA-WR3-BPB-PR-005 rev. 03, efektif 1 November 2020, berguna untuk proses tata kelola tetapi bukan bukti jadwal dan tarif 2026. Panduan EPrT Purwokerto 9–10 Mei 2025 hanya berlaku untuk kegiatan tersebut. Keduanya tidak otomatis dimasukkan ke chatbot web publik.
