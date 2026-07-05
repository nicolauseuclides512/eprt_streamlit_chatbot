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
DEFAULT_PDF_PATH = DATA_DIR / "Panduan_EPrT_LaC_Telkom_University.pdf"

DEFAULT_GENERATION_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL_CANDIDATES = ["gemini-embedding-2", "gemini-embedding-001"]
EMBEDDING_DIMENSION = 768

SYSTEM_INSTRUCTION = """
Anda adalah chatbot informasi EPrT LaC Telkom University.

Aturan utama:
1. Jawab hanya berdasarkan konteks dokumen yang diberikan.
2. Gunakan bahasa Indonesia yang sopan, jelas, dan ringkas.
3. Jangan mengarang jadwal, biaya, tautan, atau ketentuan di luar dokumen.
4. Jika informasi tidak tersedia pada konteks, katakan bahwa informasi tersebut belum tersedia di dokumen.
5. Jika ada angka, biaya, jadwal, durasi, tautan, atau kontak, tuliskan secara teliti sesuai konteks.
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


def chunk_words(text: str, chunk_size: int = 220, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end]).strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

        start = max(0, end - overlap)

    return chunks


def make_chunks(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []

    for page in pages:
        page_chunks = chunk_words(page["text"])
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
Konteks dokumen:
{context_text}

Pertanyaan pengguna:
{question}

Instruksi jawaban:
- Jawab berdasarkan konteks dokumen di atas.
- Jika informasi tidak ada di konteks, sampaikan bahwa informasi tersebut belum tersedia pada dokumen.
- Akhiri jawaban dengan bagian "Sumber" yang menyebutkan halaman dokumen yang digunakan.
""".strip()

    model = get_config_value("GEMINI_MODEL", DEFAULT_GENERATION_MODEL)

    try:
        interaction = client.interactions.create(
            model=model,
            system_instruction=SYSTEM_INSTRUCTION,
            input=prompt,
            generation_config={
                "temperature": 0.2,
            },
        )
        return interaction.output_text
    except Exception as exc:
        raise RuntimeError(
            "Gagal membuat jawaban dari Gemini. "
            f"Detail error: {exc}"
        )


# ============================================================
# Tampilan Streamlit
# ============================================================

st.set_page_config(
    page_title="Chatbot EPrT LaC",
    page_icon="E",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        :root {
            --brand: #b42318;
            --ink: #182230;
            --muted: #667085;
            --line: #e4e7ec;
        }

        .stApp {
            background: linear-gradient(180deg, #fff8e8 0%, #ffffff 34%, #f8fafc 100%);
            color: var(--ink);
        }

        [data-testid="stHeader"] {
            background: rgba(255, 248, 232, 0.78);
            backdrop-filter: blur(12px);
        }

        section[data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid var(--line);
        }

        .block-container {
            max-width: 1120px;
            padding-top: 2rem;
            padding-bottom: 6rem;
        }

        .hero {
            border: 1px solid #ffd2cc;
            border-radius: 8px;
            padding: 1.4rem 1.5rem;
            background:
                linear-gradient(135deg, rgba(255, 241, 239, 0.96), rgba(255, 248, 232, 0.92)),
                radial-gradient(circle at 82% 10%, rgba(180, 35, 24, 0.14), transparent 34%);
            box-shadow: 0 16px 40px rgba(16, 24, 40, 0.08);
            margin-bottom: 1rem;
        }

        .eyebrow {
            color: var(--brand);
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
            color: #475467;
            font-size: 1.02rem;
            margin: 0.75rem 0 0;
            max-width: 760px;
        }

        .info-strip {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 1rem 0 1.25rem;
        }

        .info-tile {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.88);
            padding: 0.85rem 1rem;
        }

        .info-tile span {
            display: block;
            color: var(--muted);
            font-size: 0.8rem;
            margin-bottom: 0.15rem;
        }

        .info-tile strong {
            color: var(--ink);
            font-size: 1rem;
        }

        .section-label {
            color: #344054;
            font-size: 0.88rem;
            font-weight: 700;
            margin: 1.1rem 0 0.45rem;
        }

        div[data-testid="stChatMessage"] {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.9);
            box-shadow: 0 8px 24px rgba(16, 24, 40, 0.05);
            padding: 0.3rem;
        }

        div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
            background: #fffaf2;
            border-color: #fedf89;
        }

        .source-card {
            border: 1px solid var(--line);
            border-left: 4px solid var(--brand);
            border-radius: 8px;
            background: #ffffff;
            padding: 0.85rem 1rem;
            margin-bottom: 0.75rem;
        }

        .source-card strong {
            color: var(--ink);
        }

        .source-card p {
            color: #475467;
            margin: 0.45rem 0 0;
            line-height: 1.55;
        }

        .stButton > button {
            border-radius: 8px;
            border-color: #fda29b;
            color: #912018;
            background: #fff7f5;
            min-height: 2.55rem;
            white-space: normal;
        }

        .stButton > button:hover {
            border-color: var(--brand);
            color: var(--brand);
            background: #fff1ef;
        }

        [data-testid="stChatInput"] {
            border-radius: 8px;
        }

        @media (max-width: 760px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .hero {
                padding: 1.1rem;
            }

            .info-strip {
                grid-template-columns: 1fr;
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
            <h1>Chatbot Panduan EPrT</h1>
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

generation_model = get_config_value("GEMINI_MODEL", DEFAULT_GENERATION_MODEL)
safe_pdf_name = html.escape(metadata["pdf_name"])

with st.sidebar:
    st.markdown("### Chatbot EPrT")
    st.caption("Asisten berbasis dokumen panduan LaC Telkom University.")

    st.markdown("#### Knowledge Base")
    st.metric("Halaman terbaca", metadata["total_pages"])
    st.metric("Potongan konteks", metadata["total_chunks"])

    st.markdown("#### Detail Sistem")
    st.write(f"Dokumen: `{metadata['pdf_name']}`")
    st.write(f"Embedding: `{metadata['embedding_model']}`")
    st.write(f"Generasi: `{generation_model}`")

    st.divider()

    if st.button("Bersihkan percakapan", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state or not st.session_state.messages:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Halo, saya chatbot panduan EPrT. "
                "Silakan tanyakan informasi tentang pendaftaran, persiapan, pelaksanaan, atau hasil tes EPrT."
            ),
        }
    ]

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">EPrT LaC Telkom University</div>
        <h1>Chatbot Panduan EPrT</h1>
        <p>Tanyakan informasi seputar pendaftaran, persiapan, pelaksanaan, hasil tes, dan ketentuan lain yang tersedia di dokumen panduan.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="info-strip">
        <div class="info-tile">
            <span>Dokumen aktif</span>
            <strong>{safe_pdf_name}</strong>
        </div>
        <div class="info-tile">
            <span>Cakupan dokumen</span>
            <strong>{metadata['total_pages']} halaman</strong>
        </div>
        <div class="info-tile">
            <span>Mode jawaban</span>
            <strong>Berbasis sumber</strong>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

suggested_questions = [
    "Bagaimana cara mendaftar tes EPrT?",
    "Apa saja yang perlu dipersiapkan sebelum tes?",
    "Bagaimana cara melihat hasil tes EPrT?",
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
                    top_k=4,
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
