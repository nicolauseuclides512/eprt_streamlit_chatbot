# LaC/EPrT Web RAG — Streamlit + Gemini + FAISS

Chatbot informasi **website resmi Language Center Telkom University**, bukan lagi chatbot berbasis PDF. Sumber web dikurasi di `sources.json`. PDF lama di folder `data/` tetap disimpan sebagai arsip, tetapi tidak dipakai sebagai knowledge base aktif.

## Arsitektur v1 (Hybrid RAG + Rule-Based FAQ + Precomputed FAISS)

1. **Preprocessing di luar Streamlit:** `scripts/build_index.py` mengambil sembilan URL resmi di `sources.json` dengan allowlist dan robots.txt, lalu menyimpan snapshot website dan embedding ke cache yang dipakai ulang.
2. **Batch embedding + checkpoint:** teks dipotong berdasarkan paragraf (950 karakter, overlap 100); embedding dibuat per kelompok 8 teks dan cache disimpan setiap batch. `gemini-embedding-2` menggunakan daftar objek `Content` agar output tidak salah digabung.
3. **Indeks FAISS pra-hitung:** file `index.faiss`, `chunks.json`, dan `metadata.json` disimpan sebagai snapshot berversi. File `current.json` hanya dipindahkan ke snapshot baru setelah berhasil.
4. **Startup cepat:** `streamlit run app.py` hanya memuat snapshot FAISS dan membuat indeks leksikal BM25 ringan dari teks. **Startup tidak melakukan crawl atau embedding dokumen lagi.**
5. **Routing FAQ:** pertanyaan yang cocok dengan aturan terkonfirmasi pada `faq_rules.json` dijawab langsung tanpa panggilan Gemini; pertanyaan lain masuk hybrid retrieval BM25 + FAISS.
6. **Validasi jawaban:** Gemini diminta menulis sitasi `[S1]` (website) atau `[P1]` (kebijakan pengelola). Sistem menolak sitasi tidak dikenal, angka yang tidak ada di bukti yang dikutip, dan tautan fiktif. **Ini pagar pengaman deterministik, bukan jaminan verifikasi semantik seluruh klaim.**
7. **Pembaruan sumber dinamis:** pertanyaan jadwal/biaya dapat mengecek satu halaman resmi secara langsung; pembaruan FAISS tetap dilakukan dengan menjalankan indexer terpisah.

Data kebijakan memuat `subject`, `predicate`, `object`, `scope`, `valid_from`, `valid_until`, `status` dan provenance agar kelak dapat dikonversi ke Knowledge Graph. `valid_from = null` berarti tanggal efektif belum ditentukan; bukan berarti kebijakan berlaku selamanya.

## Kebijakan operasional lintas kampus (konfirmasi pengelola 18 November 2026)

Kebijakan operasional berada di `institutional_policy.py` dan dibedakan dari fakta hasil crawling website:

- Jadwal, biaya, dan ketentuan kelulusan berlaku **seragam di seluruh kampus Telkom University untuk layanan dan kategori peserta yang sama**.
- Pelaksanaan **tes bahasa dan kursus bahasa di luar Bandung untuk sementara dilakukan secara daring/online**.
- Metode Bandung tetap mengacu pada pengumuman layanan resmi; kode tidak menyatakan semua layanan Bandung selalu luring.
- Program EPrT reguler dan EPrT mahasiswa baru tidak boleh digabungkan begitu saja, terutama mengenai layanan gratis.
- Tidak ada angka tarif, tanggal/slot, atau batas kelulusan numerik yang ditetapkan dalam konfirmasi ini. Angka hanya boleh disebut apabila ada dasar resmi dan berlaku.
- Aturan ini berasal dari pengelola Pusat Bahasa, **bukan kutipan halaman situs**; sumber dijelaskan terpisah pada UI.

## Menjalankan secara lokal (Windows PowerShell)

Python 3.11+, akses internet untuk proses indexing pertama, dan API key Gemini.

```powershell
cd D:\Project\eprt_streamlit_chatbot
git switch master
git pull --ff-only origin master

.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if (-not (Test-Path ".streamlit\secrets.toml")) { Copy-Item .streamlit\secrets.toml.example .streamlit\secrets.toml }
notepad .streamlit\secrets.toml

# Wajib sekali sebelum startup pertama, atau ketika website berubah
.\.venv\Scripts\python.exe scripts\build_index.py

# Setelah indeks berhasil terbentuk
.\.venv\Scripts\python.exe -m streamlit run app.py
```

**Jika `.streamlit/secrets.toml` sudah berisi API key yang valid, jangan overwrite.** Perintah `Copy-Item` di atas digunakan hanya jika belum ada file tersebut; edit dan simpan API key secara lokal, jangan bagikan.

Contoh secrets:

```toml
GEMINI_API_KEY = "API_KEY_ANDA"
GEMINI_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "gemini-embedding-001"
SOURCE_SYNC_HOURS = "12"
SOURCES_CONFIG_PATH = "sources.json"
PRECOMPUTED_INDEX_DIR = ".cache/web_rag/precomputed"
```

Untuk memperbarui semua sumber secara paksa:

```powershell
.\.venv\Scripts\python.exe scripts\build_index.py --force-refresh
```

Atau batasi batch jika terkena rate limit: `--batch-size 4`. Crawler hanya mengambil domain resmi dan mengikuti robots.txt; embedding yang tidak berubah digunakan ulang. Jangan menghapus `.cache/web_rag/embeddings.json` jika ingin melanjutkan progres.

**Penting untuk Streamlit Community Cloud:** folder `.cache/` di-`gitignore` dan tidak ikut GitHub/deployment. Indeks harus disediakan oleh proses build/deployment dan disimpan pada lokasi persisten; jika belum disediakan, aplikasi akan menampilkan perintah untuk menjalankan indexer, bukan diam-diam membuat semua embedding pada setiap startup. Integrasi artifact deployment/penjadwalan belum diimplementasikan pada v1 ini.

## Pembaruan, keamanan, dan cache

Sumber yang berhasil dicek diberi waktu pemeriksaan dan snapshot lama dipertahankan jika refresh gagal. Aplikasi hanya memuat indeks pra-hitung; untuk memperbarui pengetahuan yang sudah terindeks, **jalankan indexer lagi**. Cache embedding dan snapshot berada pada `.cache/web_rag` (abaikan dari Git). Sinkronisasi terjadwal memerlukan job dan penyimpanan persisten terpisah.

`faq_rules.json` hanya memuat kebijakan yang dikonfirmasi pengelola. Jangan menambah nominal biaya atau angka kelulusan tanpa dokumen keputusan yang telah berlaku. Data peserta/mahasiswa tidak masuk knowledge base publik.

## Kompatibilitas versi Gemini API (November 2026)

Google Gen AI Python SDK versi 3.0.0 (5 November 2026) mengubah default
Gemini Developer API dari `v1beta` menjadi `v1`. Pada chatbot ini,
`system_instruction` yang memuat kebijakan operasional dapat ditolak oleh
endpoint `v1` dengan error:

```text
400 INVALID_ARGUMENT
Developer instruction is not enabled for api version v1.
Use api version v1beta instead.
```

File `gemini_client.py` memaksa `types.HttpOptions(api_version="v1beta")`
saat membuat `genai.Client`. Client yang sama dipakai untuk pembuatan
embedding dan jawaban. Tidak perlu menonaktifkan `system_instruction`,
mengubah API key, atau melakukan downgrade SDK untuk memperbaiki error ini.
Restart Streamlit setelah `git pull` dan ulangi pertanyaan.

## Diagnosis gagal menjawab

Jika jawaban Gemini gagal dibuat, chatbot **tidak menampilkan tautan hasil pencarian sebagai pengganti jawaban**. Aplikasi menampilkan tahap yang gagal beserta kategori error yang aman tanpa menampilkan isi API key/prompt dari exception.

| Indikator | Arti | Perbaikan pertama |
|---|---|---|
| `API_KEY_INVALID` | Kunci API salah atau tidak aktif | Periksa `.streamlit/secrets.toml`; jangan bagikan key |
| `429 RESOURCE_EXHAUSTED` | Kuota/rate limit Google tercapai | Periksa kuota Google AI Studio, lalu tunggu batasnya tersedia |
| `404 NOT_FOUND` | Model generasi tidak tersedia bagi key | Periksa `GEMINI_MODEL` |
| `400 INVALID_ARGUMENT` | Permintaan/parameter model tidak diterima | Periksa log terminal dan kompatibilitas SDK/model |
| `503 UNAVAILABLE` | Layanan model sementara tidak tersedia | Ulangi setelah beberapa saat |
| `timeout` / koneksi | Gagal menjangkau API/website | Periksa jaringan/proxy/firewall |

Log exception lengkap dicetak di **terminal yang menjalankan Streamlit**, bukan di chat. Jangan kirim log yang masih memuat API key atau data pribadi. **Keberhasilan sinkronisasi halaman web bukan bukti jawaban Gemini berhasil.** Verifikasi kode dengan `python -m unittest discover -s tests -v`; pengujian end-to-end tetap memerlukan Gemini API key yang valid.

## Test

~~~bash
python -m unittest discover -s tests -v
python -m compileall -q app.py rag_pipeline.py faq_rules.py answer_guard.py scripts tests
~~~

GitHub Actions menjalankan test unit secara offline; tidak melakukan panggilan Gemini atau website produksi. Deployment, kualitas jawaban, hak akses sumber dan kuota API tetap harus diuji manual sebelum go-live.

**Batasan:** ini chatbot informasi publik, tidak mengakses nilai tes atau data pribadi. Isi website diperlakukan sebagai data, bukan instruksi. Informasi jadwal, biaya dan syarat resmi tetap perlu dikonfirmasi pada tautan sumber.
