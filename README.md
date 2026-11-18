# LaC/EPrT Web RAG — Streamlit + Gemini + FAISS

Chatbot informasi **website resmi Language Center Telkom University**, bukan lagi chatbot berbasis PDF. Sumber web dikurasi di `sources.json». PDF lama di folder `data/» tetap disimpan sebagai arsip, tetapi tidak dipakai sebagai knowledge base aktif.

## Arsitektur

1. Hanya URL HTTPS dari domain yang secara eksplisit disetujui di `web_knowledge.py». URL yang dipakai ada dalam `sources.json»; tidak ada crawling tautan bebas.
2. Cek robots.txt, batasi redirect, respons dan waktu tunggu; ekstraksi HTML dengan Trafilatura dan fallback BeautifulSoup.
3. Simpan snapshot HTML-terekstraksi, digest, sumber dan waktu pemeriksaan pada `.cache/web_rag/pages.json».
4. Pecah menjadi chunk dan simpan embedding per hash di `.cache/web_rag/embeddings.json». Hanya chunk baru/berubah yang di-embedding ulang. Cari chunk dengan FAISS.
5. Gemini menjawab berdasarkan konteks terpilih; Streamlit menampilkan URL sumber dan status pemeriksaan. Pertanyaan tentang jadwal, biaya, kuota, tanggal, atau pendaftaran meminta satu pemeriksaan langsung pada halaman dinamis yang relevan.

Jika halaman tidak dapat diambil, snapshot sebelumnya dipertahankan dan UI memperlihatkan peringatan. Cakupan `campus» tercatat agar aturan Bandung tidak otomatis diterapkan untuk Purwokerto, Jakarta, Surabaya atau PJJ.

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

Edit `.streamlit/secrets.toml» sebelum menjalankan aplikasi:

~~~toml
GEMINI_API_KEY = "API_KEY_ANDA"
GEMINI_MODEL = "gemini-2.5-flash"
SOURCE_SYNC_HOURS = "12"
SOURCES_CONFIG_PATH = "sources.json"
~~~

Pada Windows, aktivasi environment dapat menggunakan `.venv\Scripts\activate». Jangan unggah API key atau secrets ke GitHub. Streamlit Community Cloud menggunakan `app.py» sebagai main file dan Secrets di dashboard cloud.

## Pembaruan sumber dan cache

Sumber yang berumur minimal `SOURCE_SYNC_HOURS» (default 12 jam) akan dicek ulang ketika aplikasi aktif. Cache resource Streamlit dievaluasi setiap 30 menit pada akses/rerun. **Ini bukan scheduler latar 24/7** dan tidak menjamin pembaruan saat aplikasi tidur. Pada Streamlit Community Cloud, direktori cache lokal tidak persisten setelah rebuild/restart. Untuk sinkronisasi tanpa pengguna, perlu penjadwal dan penyimpanan persisten yang disiapkan terpisah.

Ubah `sources.json» untuk menambah/mengurangi halaman, dengan tetap membatasi pada halaman publik resmi dan mengikuti robots.txt. Untuk domain baru, ubah allowlist di kode setelah pemeriksaan operator. `dynamic: true» mengaktifkan pemeriksaan live untuk pertanyaan sensitif waktu.

## Test

~~~bash
python -m unittest discover -s tests -v
python -m compileall -q app.py web_knowledge.py
~~~

GitHub Actions menjalankan test unit secara offline; tidak melakukan panggilan Gemini atau website produksi. Deployment, kualitas jawaban, hak akses sumber dan kuota API tetap harus diuji manual sebelum go-live.

**Batasan:** ini chatbot informasi publik, tidak mengakses nilai tes atau data pribadi. Isi website diperlakukan sebagai data, bukan instruksi. Informasi jadwal, biaya dan syarat resmi tetap perlu dikonfirmasi pada tautan sumber.
