# Chatbot RAG EPrT LaC dengan Streamlit + Gemini

Project ini adalah versi Python lokal dari notebook Colab Streamlit. Aplikasi dibuat untuk dijalankan dengan Streamlit dan dapat di-deploy melalui GitHub ke Streamlit Community Cloud.

Chatbot menggunakan pendekatan RAG (Retrieval-Augmented Generation):
1. Membaca dokumen PDF dari folder `data/`.
2. Mengekstrak teks dari PDF.
3. Memecah teks menjadi chunk.
4. Membuat embedding dengan Gemini.
5. Mencari chunk yang paling relevan dengan FAISS.
6. Mengirim konteks dokumen ke Gemini untuk menghasilkan jawaban.

## Struktur folder

```text
eprt_streamlit_chatbot/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── secrets.toml.example
└── data/
    └── Panduan_EPrT_LaC_Telkom_University.pdf
```

## Menjalankan di komputer lokal

Buat virtual environment:

```bash
python -m venv .venv
```

Aktifkan virtual environment.

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install library:

```bash
pip install -r requirements.txt
```

Siapkan API key.

Opsi 1, menggunakan environment variable:

Windows PowerShell:

```powershell
$env:GEMINI_API_KEY="ISI_API_KEY_ANDA"
```

macOS/Linux:

```bash
export GEMINI_API_KEY="ISI_API_KEY_ANDA"
```

Opsi 2, menggunakan file Streamlit Secrets lokal:

```bash
mkdir .streamlit
copy .streamlit\secrets.toml.example .streamlit\secrets.toml
```

Untuk macOS/Linux:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Lalu edit `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "ISI_API_KEY_ANDA"
GEMINI_MODEL = "gemini-2.5-flash"
```

Jalankan aplikasi:

```bash
streamlit run app.py
```

## Deploy ke Streamlit Community Cloud lewat GitHub

1. Buat repository GitHub baru.
2. Upload semua file project ini ke repository.
3. Pastikan file `.streamlit/secrets.toml` tidak ikut di-upload.
4. Buka Streamlit Community Cloud.
5. Pilih repository GitHub.
6. Main file path: `app.py`.
7. Tambahkan secrets pada menu app settings:

```toml
GEMINI_API_KEY = "ISI_API_KEY_ANDA"
GEMINI_MODEL = "gemini-2.5-flash"
```

8. Deploy.

## Mengganti dokumen PDF

Ganti file PDF di folder `data/`, lalu sesuaikan nama file pada `PDF_PATH` di Secrets jika namanya berbeda:

```toml
PDF_PATH = "data/nama_dokumen_baru.pdf"
```

Jika hanya ada satu file PDF di folder `data/`, aplikasi akan mencoba membaca file PDF tersebut secara otomatis.

## Catatan keamanan

Jangan commit file `.streamlit/secrets.toml` ke GitHub karena berisi API key. File ini sudah dimasukkan ke `.gitignore`.

Jika dokumen knowledge base bersifat internal atau rahasia, jangan upload dokumen tersebut ke repository publik.
