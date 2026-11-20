# LaC/EPrT Web RAG — Streamlit + Gemini + FAISS

Chatbot informasi **website resmi Language Center Telkom University**, bukan lagi chatbot berbasis PDF. Sumber web dikurasi di `sources.json`. PDF lama di folder `data/` tetap disimpan sebagai arsip, tetapi tidak dipakai sebagai knowledge base aktif.

## Arsitektur

1. Hanya URL HTTPS dari domain yang secara eksplisit disetujui di `web_knowledge.py`. URL yang dipakai ada dalam `sources.json`; tidak ada crawling tautan bebas.
2. Cek robots.txt, batasi redirect, respons dan waktu tunggu; ekstraksi HTML dengan Trafilatura dan fallback BeautifulSoup.
3. Simpan snapshot HTML-terekstraksi, digest, sumber dan waktu pemeriksaan pada `.cache/web_rag/pages.json`.
4. Pecah menjadi chunk dan simpan embedding per hash di `.cache/web_rag/embeddings.json`. Hanya chunk baru/berubah yang di-embedding ulang. Cari chunk dengan FAISS.
5. Gemini menjawab berdasarkan konteks terpilih; Streamlit menampilkan URL sumber dan status pemeriksaan. Pertanyaan tentang jadwal, biaya, kuota, tanggal, atau pendaftaran meminta satu pemeriksaan langsung pada halaman dinamis yang relevan.

Jika halaman tidak dapat diambil, snapshot sebelumnya dipertahankan dan UI memperlihatkan peringatan. Metadata `campus` tetap digunakan untuk mengidentifikasi cakupan pengumuman atau program tertentu, **bukan** untuk mengasumsikan perbedaan jadwal/biaya/kelulusan per kampus.


## Kebijakan operasional lintas kampus (konfirmasi pengelola 18 November 2026)

Kebijakan operasional berada di `institutional_policy.py` dan dibedakan dari fakta hasil crawling website:

- Jadwal, biaya, dan ketentuan kelulusan berlaku **seragam di seluruh kampus Telkom University untuk layanan dan kategori peserta yang sama**.
- Pelaksanaan **tes bahasa dan kursus bahasa di luar Bandung untuk sementara dilakukan secara daring/online**.
- Metode Bandung tetap mengacu pada pengumuman layanan resmi; kode tidak menyatakan semua layanan Bandung selalu luring.
- Program EPrT reguler dan EPrT mahasiswa baru tidak boleh digabungkan begitu saja, terutama mengenai layanan gratis.
- Tidak ada angka tarif, tanggal/slot, atau batas kelulusan numerik yang ditetapkan dalam konfirmasi ini. Angka hanya boleh disebut apabila ada dasar resmi dan berlaku.
- Aturan ini berasal dari pengelola Pusat Bahasa, **bukan kutipan halaman situs**; sumber dijelaskan terpisah pada UI.

## Menjalankan secara lokal

Memerlukan Python 3.11+, internet dan API key Gemini.

~~~bash
git clone https://github.com/nicolauseuclides512/eprt_streamlit_chatbot.git
cd eprt_streamlit_chatbot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
streamlit run app.py
~~~

Edit `.streamlit/secrets.toml` sebelum menjalankan aplikasi:

~~~toml
GEMINI_API_KEY = "API_KEY_ANDA"
GEMINI_MODEL = "gemini-2.5-flash"
SOURCE_SYNC_HOURS = "12"
SOURCES_CONFIG_PATH = "sources.json"
~~~

Pada Windows, aktivasi environment dapat menggunakan `.venv\Scripts\activate`. Jangan unggah API key atau secrets ke GitHub. Streamlit Community Cloud menggunakan `app.py` sebagai main file dan Secrets di dashboard cloud.

## Pembaruan sumber dan cache

Sumber yang berumur minimal `SOURCE_SYNC_HOURS` (default 12 jam) akan dicek ulang ketika aplikasi aktif. Cache resource Streamlit dievaluasi setiap 30 menit pada akses/rerun. **Ini bukan scheduler latar 24/7** dan tidak menjamin pembaruan saat aplikasi tidur. Pada Streamlit Community Cloud, direktori cache lokal tidak persisten setelah rebuild/restart. Untuk sinkronisasi tanpa pengguna, perlu penjadwal dan penyimpanan persisten yang disiapkan terpisah.

Ubah `sources.json` untuk menambah/mengurangi halaman, dengan tetap membatasi pada halaman publik resmi dan mengikuti robots.txt. Untuk domain baru, ubah allowlist di kode setelah pemeriksaan operator. `dynamic: true` mengaktifkan pemeriksaan live untuk pertanyaan sensitif waktu.

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
python -m compileall -q app.py web_knowledge.py
~~~

GitHub Actions menjalankan test unit secara offline; tidak melakukan panggilan Gemini atau website produksi. Deployment, kualitas jawaban, hak akses sumber dan kuota API tetap harus diuji manual sebelum go-live.

**Batasan:** ini chatbot informasi publik, tidak mengakses nilai tes atau data pribadi. Isi website diperlakukan sebagai data, bukan instruksi. Informasi jadwal, biaya dan syarat resmi tetap perlu dikonfirmasi pada tautan sumber.
