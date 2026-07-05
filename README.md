# Chatbot RAG LaC Telkom University dengan Streamlit + Gemini

Project ini adalah aplikasi chatbot berbasis Streamlit untuk menjawab pertanyaan dari dokumen kompilasi informasi website Telkom University Language Center (LaC).

Chatbot menggunakan pendekatan RAG (Retrieval-Augmented Generation):
1. Membaca PDF dari folder `data/`.
2. Mengekstrak teks dari PDF.
3. Memecah teks menjadi chunk dengan overlap seperti pola `RecursiveCharacterTextSplitter`.
4. Membuat embedding dengan Gemini.
5. Mencari chunk paling relevan menggunakan FAISS.
6. Mengirim konteks dokumen ke Gemini untuk menghasilkan jawaban berbasis sumber.

## Struktur Folder

```text
eprt_streamlit_chatbot/
|-- app.py
|-- requirements.txt
|-- README.md
|-- .gitignore
|-- .streamlit/
|   |-- config.toml
|   `-- secrets.toml.example
`-- data/
    |-- Kompilasi_Informasi_Website_LaC_Telkom_University.pdf
    `-- Panduan_EPrT_LaC_Telkom_University.pdf
```

## Menjalankan Lokal

Buat dan aktifkan virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependency:

```bash
pip install -r requirements.txt
```

Salin contoh secrets:

```bash
copy .streamlit\secrets.toml.example .streamlit\secrets.toml
```

Isi `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "ISI_API_KEY_ANDA"
GEMINI_MODEL = "gemini-2.5-flash"
PDF_PATH = "data/Kompilasi_Informasi_Website_LaC_Telkom_University.pdf"
```

Jalankan aplikasi:

```bash
streamlit run app.py
```

## Deploy Streamlit Community Cloud

1. Upload project ke GitHub.
2. Pastikan `.streamlit/secrets.toml` tidak ikut di-upload.
3. Buat app baru di Streamlit Community Cloud.
4. Set main file path ke `app.py`.
5. Tambahkan secrets yang sama seperti contoh di atas pada menu app settings.
6. Deploy.

## Mengganti Dokumen PDF

Jika ingin memakai dokumen lain, simpan PDF di folder `data/`, lalu ubah nilai `PDF_PATH` di secrets:

```toml
PDF_PATH = "data/nama_dokumen_baru.pdf"
```

## Catatan Keamanan

Jangan commit file `.streamlit/secrets.toml` ke GitHub karena berisi API key.
