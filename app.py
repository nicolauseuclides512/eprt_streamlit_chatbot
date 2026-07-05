import html
import os
import re
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader


# ============================================================
# Konfigurasi aplikasi
# ============================================================

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
DEFAULT_PDF_PATH = DATA_DIR / "Kompilasi_Informasi_Website_LaC_Telkom_University.pdf"

DEFAULT_GENERATION_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL_CANDIDATES = ["gemini-embedding-2", "gemini-embedding-001"]
EMBEDDING_DIMENSION = 768
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
RETRIEVAL_TOP_K = 8

SYSTEM_INSTRUCTION = """
Anda adalah chatbot informasi Telkom University Language Center (LaC).

Aturan utama:
1. Jawab hanya berdasarkan konteks dokumen kompilasi website LaC yang diberikan.
2. Gunakan bahasa Indonesia yang sopan, jelas, dan ringkas.
3. Jangan mengarang jadwal, biaya, tautan, kontak, persyaratan, atau ketentuan di luar dokumen.
4. Jika informasi tidak tersedia pada konteks, katakan bahwa jawaban tidak tersedia dalam konteks dokumen.
5. Jika ada angka, tanggal, biaya, jadwal, durasi, tautan, atau kontak, tuliskan secara teliti sesuai konteks.
6. Jika konteks memuat beberapa layanan yang mirip, jelaskan perbedaannya agar pengguna tidak tertukar.
""".strip()


# ============================================================
# Utilitas konfigurasi backend
# ============================================================

def get_config_value(name: str, default: str | None = None) -> str | None:
    """
    Mengambil konfigurasi dari Streamlit Secrets terlebih dahulu,
    lalu fallback ke environment variable.

    Untuk Streamlit Community Cloud:
    - buka menu Settings > Secrets
    - isi:
      GEMINI_API_KEY = "isi_api_key_anda"
    """
    try:
        value = st.secrets.get(name)
        if value:
            return str(value)
    except Exception:
        pass

    value = os.getenv(name)
    if value:
        return value

    return default


def get_pdf_path() -> Path:
    """
    Dokumen knowledge base disimpan di backend/repository.
    Tidak ada fitur upload dokumen dari UI.
    """
    configured_path = get_config_value("PDF_PATH")
    if configured_path:
        path = Path(configured_path)
        if not path.is_absolute():
            path = APP_DIR / configured_path
        return path

    if DEFAULT_PDF_PATH.exists():
        return DEFAULT_PDF_PATH

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    if pdf_files:
        return pdf_files[0]

    return DEFAULT_PDF_PATH


# ============================================================
# PDF, chunking, embedding, dan retrieval
# ============================================================

def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_pdf_pages(pdf_path: Path) -> list[dict[str, Any]]:
    reader = PdfReader(str(pdf_path))
    pages: list[dict[str, Any]] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = clean_text(text)
        if text:
            pages.append({
                "page": page_number,
                "text": text,
            })

    if not pages:
        raise ValueError(
            "Tidak ada teks yang berhasil diekstrak dari PDF. "
            "Pastikan PDF berisi teks digital, bukan scan gambar murni."
        )

    return pages


def split_with_overlap(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(0, end - overlap)

    return chunks


def recursive_chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Versi ringan dari pola RecursiveCharacterTextSplitter pada notebook:
    pecah berdasarkan paragraf/kalimat lebih dulu, lalu fallback ke panjang karakter.
    """
    separators = ["\n\n", "\n", ". ", "; ", ", ", " "]

    def split_recursive(segment: str, separator_index: int = 0) -> list[str]:
        segment = segment.strip()
        if not segment:
            return []
        if len(segment) <= chunk_size:
            return [segment]
        if separator_index >= len(separators):
            return split_with_overlap(segment, chunk_size, overlap)

        separator = separators[separator_index]
        parts = [part.strip() for part in segment.split(separator) if part.strip()]
        if len(parts) == 1:
            return split_recursive(segment, separator_index + 1)

        chunks: list[str] = []
        current = ""

        for part in parts:
            candidate = f"{current}{separator}{part}".strip() if current else part
            if len(candidate) <= chunk_size:
                current = candidate
                continue

            if current:
                chunks.extend(split_recursive(current, separator_index + 1))
            current = part

        if current:
            chunks.extend(split_recursive(current, separator_index + 1))

        if overlap <= 0 or len(chunks) <= 1:
            return chunks

        overlapped: list[str] = []
        previous_tail = ""
        for chunk in chunks:
            combined = f"{previous_tail} {chunk}".strip() if previous_tail else chunk
            overlapped.append(combined[: chunk_size + overlap].strip())
            previous_tail = chunk[-overlap:]

        return overlapped

    return split_recursive(text)


def make_chunks(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []

    for page in pages:
        page_chunks = recursive_chunk_text(page["text"])
        for chunk_number, chunk_text in enumerate(page_chunks, start=1):
            chunks.append({
                "id": len(chunks),
                "page": page["page"],
                "chunk_no": chunk_number,
                "text": chunk_text,
            })

    return chunks


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    vectors = vectors.astype("float32")
    faiss.normalize_L2(vectors)
    return vectors


def embed_one_text(client: genai.Client, text: str, model_name: str, task_type: str) -> list[float]:
    """
    Embedding dibuat per teks agar kompatibel dengan gemini-embedding-2
    dan fallback gemini-embedding-001.
    """
    if model_name == "gemini-embedding-2":
        config = types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIMENSION)
    else:
        config = types.EmbedContentConfig(task_type=task_type)

    result = client.models.embed_content(
        model=model_name,
        contents=text,
        config=config,
    )

    return result.embeddings[0].values


def choose_embedding_model(client: genai.Client) -> str:
    last_error: Exception | None = None

    for model_name in EMBEDDING_MODEL_CANDIDATES:
        try:
            _ = embed_one_text(
                client=client,
                text="uji model embedding",
                model_name=model_name,
                task_type="RETRIEVAL_DOCUMENT",
            )
            return model_name
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        "Tidak ada model embedding yang dapat digunakan. "
        f"Error terakhir: {last_error}"
    )


def embed_texts(
    client: genai.Client,
    texts: list[str],
    model_name: str,
    task_type: str,
    delay_seconds: float = 0.15,
) -> np.ndarray:
    vectors = []

    for text in texts:
        vectors.append(
            embed_one_text(
                client=client,
                text=text,
                model_name=model_name,
                task_type=task_type,
            )
        )
        time.sleep(delay_seconds)

    return normalize_vectors(np.array(vectors, dtype="float32"))


@st.cache_resource(show_spinner=False)
def build_rag_backend(api_key: str, pdf_path_str: str):
    """
    Backend RAG dibuat sekali saat aplikasi berjalan.
    Hasilnya di-cache oleh Streamlit agar tidak membuat ulang embedding setiap rerun.
    """
    pdf_path = Path(pdf_path_str)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"File PDF tidak ditemukan: {pdf_path}. "
            "Pastikan dokumen tersedia di folder data/ atau atur PDF_PATH di secrets."
        )

    client = genai.Client(api_key=api_key)

    pages = extract_pdf_pages(pdf_path)
    chunks = make_chunks(pages)

    if not chunks:
        raise ValueError("Chunk dokumen kosong. Periksa isi PDF.")

    embedding_model = choose_embedding_model(client)
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_texts(
        client=client,
        texts=texts,
        model_name=embedding_model,
        task_type="RETRIEVAL_DOCUMENT",
    )

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    metadata = {
        "pdf_name": pdf_path.name,
        "total_pages": len(pages),
        "total_chunks": len(chunks),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "embedding_model": embedding_model,
        "embedding_dimension": int(embeddings.shape[1]),
    }

    return client, index, chunks, metadata


def embed_query(client: genai.Client, query: str, embedding_model: str) -> np.ndarray:
    vector = embed_one_text(
        client=client,
        text=query,
        model_name=embedding_model,
        task_type="RETRIEVAL_QUERY",
    )
    return normalize_vectors(np.array([vector], dtype="float32"))


def retrieve_contexts(
    client: genai.Client,
    index: faiss.IndexFlatIP,
    chunks: list[dict[str, Any]],
    metadata: dict[str, Any],
    query: str,
    top_k: int = 4,
) -> list[dict[str, Any]]:
    query_vector = embed_query(client, query, metadata["embedding_model"])
    scores, ids = index.search(query_vector, top_k)

    results: list[dict[str, Any]] = []

    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue

        item = dict(chunks[int(idx)])
        item["score"] = float(score)
        results.append(item)

    return results


def generate_answer(
    client: genai.Client,
    question: str,
    contexts: list[dict[str, Any]],
) -> str:
    context_text = "\n\n".join(
        f"[Sumber: halaman {ctx['page']}, chunk {ctx['chunk_no']}]\n{ctx['text']}"
        for ctx in contexts
    )

    prompt = f"""
Konteks dokumen kompilasi website LaC Telkom University:
{context_text}

Pertanyaan pengguna:
{question}

Instruksi jawaban:
- Jawablah pertanyaan dengan sejelas dan sedetail mungkin berdasarkan konteks yang diberikan.
- Jika jawaban tidak ada dalam konteks, cukup katakan: "Jawaban tidak tersedia dalam konteks dokumen."
- Jangan memberikan jawaban yang salah atau mengarang informasi di luar konteks.
- Gunakan poin-poin ringkas jika jawabannya berisi daftar layanan, jadwal, syarat, kontak, atau langkah.
- Akhiri jawaban dengan bagian "Sumber" yang menyebutkan halaman dokumen yang digunakan.
""".strip()

    model = get_config_value("GEMINI_MODEL", DEFAULT_GENERATION_MODEL)

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.2,
                top_p=0.95,
                top_k=20,
            ),
        )
        return response.text or "Jawaban tidak tersedia dalam konteks dokumen."
    except Exception as exc:
        raise RuntimeError(
            "Gagal membuat jawaban dari Gemini. "
            f"Detail error: {exc}"
        )


# ============================================================
# Tampilan Streamlit
# ============================================================

st.set_page_config(
    page_title="Chatbot LaC Telkom University",
    page_icon="E",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        :root {
            --brand: #d71920;
            --brand-deep: #8f1117;
            --gold: #f5b335;
            --ink: #f8fafc;
            --muted: #a6b0c3;
            --line: rgba(255, 255, 255, 0.12);
            --panel: rgba(15, 23, 42, 0.88);
            --panel-soft: rgba(30, 41, 59, 0.74);
        }

        .stApp {
            background:
                linear-gradient(180deg, #090d18 0%, #101827 48%, #121826 100%);
            color: var(--ink);
        }

        [data-testid="stHeader"] {
            background: rgba(9, 13, 24, 0.78);
            backdrop-filter: blur(12px);
        }

        section[data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, #111827 0%, #0b1120 100%);
            border-right: 1px solid var(--line);
        }

        .block-container {
            max-width: 1120px;
            padding-top: 2rem;
            padding-bottom: 6rem;
        }

        .hero {
            border: 1px solid rgba(245, 179, 53, 0.32);
            border-radius: 8px;
            padding: 1.4rem 1.5rem;
            background:
                linear-gradient(135deg, rgba(143, 17, 23, 0.92), rgba(17, 24, 39, 0.94) 58%),
                radial-gradient(circle at 84% 12%, rgba(245, 179, 53, 0.24), transparent 34%);
            box-shadow: 0 18px 48px rgba(0, 0, 0, 0.28);
            margin-bottom: 1rem;
        }

        .eyebrow {
            color: var(--gold);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }

        .hero h1 {
            color: var(--ink);
            font-size: clamp(2rem, 4vw, 3.15rem);
            line-height: 1.05;
            margin: 0;
        }

        .hero p {
            color: #d7dde8;
            font-size: 1.02rem;
            margin: 0.75rem 0 0;
            max-width: 760px;
        }

        .section-label {
            color: #f4f7fb;
            font-size: 0.88rem;
            font-weight: 700;
            margin: 1.1rem 0 0.45rem;
        }

        div[data-testid="stChatMessage"] {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: var(--panel);
            box-shadow: 0 10px 28px rgba(0, 0, 0, 0.22);
            padding: 0.3rem;
        }

        div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
            background: rgba(143, 17, 23, 0.42);
            border-color: rgba(245, 179, 53, 0.38);
        }

        .source-card {
            border: 1px solid var(--line);
            border-left: 4px solid var(--gold);
            border-radius: 8px;
            background: var(--panel-soft);
            padding: 0.85rem 1rem;
            margin-bottom: 0.75rem;
        }

        .source-card strong {
            color: var(--ink);
        }

        .source-card p {
            color: #cbd5e1;
            margin: 0.45rem 0 0;
            line-height: 1.55;
        }

        div[data-testid="stExpander"] {
            border-color: var(--line);
            background: rgba(15, 23, 42, 0.66);
            border-radius: 8px;
        }

        .stButton > button {
            border-radius: 8px;
            border-color: rgba(245, 179, 53, 0.42);
            color: #fff7d6;
            background: linear-gradient(135deg, rgba(143, 17, 23, 0.92), rgba(96, 16, 20, 0.86));
            min-height: 2.55rem;
            white-space: normal;
        }

        .stButton > button:hover {
            border-color: var(--gold);
            color: #ffffff;
            background: linear-gradient(135deg, var(--brand), var(--brand-deep));
        }

        [data-testid="stChatInput"] {
            border-radius: 8px;
        }

        [data-testid="stChatInput"] textarea {
            background: #0f172a;
            color: var(--ink);
            border-color: rgba(245, 179, 53, 0.26);
        }

        .stMarkdown, .stCaption, p, li, label {
            color: #dce3ee;
        }

        h1, h2, h3, h4, h5, h6 {
            color: #ffffff;
        }

        @media (max-width: 760px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .hero {
                padding: 1.1rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

api_key = get_config_value("GEMINI_API_KEY")
pdf_path = get_pdf_path()

if not api_key:
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">Konfigurasi dibutuhkan</div>
            <h1>Chatbot Informasi LaC</h1>
            <p>GEMINI_API_KEY belum tersedia di backend. Tambahkan GEMINI_API_KEY di Streamlit Secrets atau environment variable.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

try:
    with st.spinner("Menyiapkan knowledge base dari dokumen panduan..."):
        client, index, chunks, metadata = build_rag_backend(api_key, str(pdf_path))
except Exception as exc:
    st.error(str(exc))
    st.stop()

with st.sidebar:
    st.markdown("### Chatbot LaC")
    st.caption("Asisten berbasis kompilasi informasi website LaC Telkom University.")
    st.divider()

    if st.button("Bersihkan percakapan", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state or not st.session_state.messages:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Halo, saya chatbot informasi LaC Telkom University. "
                "Silakan tanyakan informasi tentang layanan tes bahasa, kursus, terjemahan, FAQ, kontak, atau informasi lain yang tersedia di dokumen."
            ),
        }
    ]

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Telkom University Language Center</div>
        <h1>Chatbot Informasi LaC</h1>
        <p>Tanyakan informasi seputar layanan tes bahasa, kursus bahasa, terjemahan, FAQ, kontak, berita, dan layanan lain yang tersedia di dokumen kompilasi website LaC.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

suggested_questions = [
    "Apa saja layanan yang tersedia di LaC?",
    "Bagaimana informasi tes EPrT di LaC?",
    "Apa kontak atau kanal resmi LaC?",
]

if "draft_question" not in st.session_state:
    st.session_state.draft_question = ""

st.markdown('<div class="section-label">Contoh pertanyaan</div>', unsafe_allow_html=True)
cols = st.columns(len(suggested_questions))
for col, sample in zip(cols, suggested_questions):
    with col:
        if st.button(sample, use_container_width=True):
            st.session_state.draft_question = sample

st.markdown('<div class="section-label">Percakapan</div>', unsafe_allow_html=True)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Tulis pertanyaan Anda di sini...")
if not question and st.session_state.draft_question:
    question = st.session_state.draft_question
    st.session_state.draft_question = ""

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        contexts = []

        with st.spinner("Mencari jawaban paling relevan dari dokumen..."):
            try:
                contexts = retrieve_contexts(
                    client=client,
                    index=index,
                    chunks=chunks,
                    metadata=metadata,
                    query=question,
                    top_k=RETRIEVAL_TOP_K,
                )
                answer = generate_answer(
                    client=client,
                    question=question,
                    contexts=contexts,
                )
            except Exception as exc:
                answer = f"Maaf, terjadi kendala saat memproses pertanyaan. Detail: {exc}"

            st.markdown(answer)

            with st.expander("Lihat potongan dokumen yang digunakan", expanded=False):
                if contexts:
                    for ctx in contexts:
                        preview = ctx["text"][:900]
                        if len(ctx["text"]) > 900:
                            preview += "..."
                        safe_preview = html.escape(preview)
                        st.markdown(
                            f"""
                            <div class="source-card">
                                <strong>Halaman {ctx['page']} - skor {ctx['score']:.3f}</strong>
                                <p>{safe_preview}</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                else:
                    st.write("Tidak ada konteks yang ditemukan.")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
    })
