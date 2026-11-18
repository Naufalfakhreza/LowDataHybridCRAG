import streamlit as st
import time
from typing import Optional

from RAG import BPAHybridCRAG, detect_prodi
from config import get_installed_ollama_models, OLLAMA_MODEL, stream_with_ollama

# Page configuration
st.set_page_config(
    page_title="Asisten Akademik FMIPA UNJ 2026",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .status-badge-correct {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    .status-badge-ambiguous {
        background-color: #FEF08A;
        color: #713F12;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    .status-badge-incorrect {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Memuat indeks korpus FMIPA UNJ (BM25 + Dense + Cross-Encoder)...")
def load_rag_engine():
    return BPAHybridCRAG(
        chunks_jsonl_path="bpa_corpus_chunks.jsonl",
    )


# Initialize RAG Engine
try:
    rag = load_rag_engine()
except Exception as e:
    st.error(f"Gagal memuat engine RAG: {e}")
    st.stop()

# Available study programs
PRODI_OPTIONS = [
    "Otomatis (Deteksi dari pertanyaan)",
    "Semua Program Studi (Umum)",
    "Pendidikan Matematika",
    "Program Studi S-2 Pendidikan Matematika",
    "Matematika",
    "Pendidikan Biologi",
    "Program Studi S-2 Pendidikan Biologi",
    "Biologi",
    "Pendidikan Fisika",
    "Program Studi S-2 Pendidikan Fisika",
    "Fisika",
    "Pendidikan Kimia",
    "Program Studi S-2 Pendidikan Kimia",
    "Kimia",
    "Ilmu Komputer",
    "Statistika",
]

# Sidebar controls
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/0/07/Logo_Universitas_Negeri_Jakarta.png",
        width=90,
    )
    st.title("⚙️ Pengaturan Chatbot")

    installed_models = get_installed_ollama_models()
    chat_models = [m for m in installed_models if "embed" not in m.lower()] or [OLLAMA_MODEL]
    default_idx = chat_models.index(OLLAMA_MODEL) if OLLAMA_MODEL in chat_models else 0

    selected_model = st.selectbox(
        "Model LLM (Ollama Lokal):",
        options=chat_models,
        index=default_idx,
        help="Model Ollama lokal yang digunakan untuk inferensi.",
    )

    selected_prodi = st.selectbox(
        "Filter Program Studi:",
        options=PRODI_OPTIONS,
        index=0,
        help="Pilih spesifik program studi atau biarkan Otomatis mendeteksi dari pertanyaan.",
    )

    top_k = st.slider(
        "Retrieval Candidates (top_k):",
        min_value=5,
        max_value=25,
        value=10,
        step=1,
        help="Jumlah dokumen kandidat teratas yang diambil sebelum dievaluasi Cross-Encoder.",
    )

    st.markdown("---")
    st.markdown(f"**Status Sistem:**")
    st.markdown(f"- 📚 Dokumen Korpus: `{len(rag.corpus)} chunks`")
    st.markdown(f"- 🧠 Model Aktif: `{selected_model}`")
    st.markdown(f"- ⚡ Reranker: `bge-reranker-v2-m3`")

    if st.button("🧹 Bersihkan Riwayat Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main Header
st.markdown('<div class="main-header">🎓 Asisten Akademik FMIPA UNJ 2026</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Tanya jawab interaktif berbasis Corrective RAG (CRAG) untuk Buku Pedoman Akademik FMIPA UNJ 2026/2027.</div>',
    unsafe_allow_html=True,
)

# Initialize Session Messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Halo! Saya adalah asisten akademik Fakultas MIPA Universitas Negeri Jakarta.\n\n"
                "Anda dapat menanyakan hal-hal seputar:\n"
                "- **Kurikulum & Sebaran Mata Kuliah** (misal: SKS, semester pengambilan, prasyarat skripsi)\n"
                "- **Profil Lulusan & Capaian Pembelajaran (CPL)**\n"
                "- **Status Akreditasi Program Studi**\n"
                "- **Tata Kelola & Panduan Akademik FMIPA**\n\n"
                "Ada yang bisa saya bantu hari ini?"
            ),
            "meta": None,
        }
    ]

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("meta"):
            meta = msg["meta"]
            status = meta.get("status", "CORRECT")
            conf = meta.get("conf", 0.0)
            docs = meta.get("docs", [])
            strips = meta.get("strips", "")

            badge_class = (
                "status-badge-correct"
                if status == "CORRECT"
                else ("status-badge-ambiguous" if status == "AMBIGUOUS" else "status-badge-incorrect")
            )
            st.markdown(
                f'<span class="{badge_class}">CRAG: {status} (Keyakinan: {conf:.2f})</span>',
                unsafe_allow_html=True,
            )

            with st.expander("🔍 Lihat Dokumen Sitasi & Knowledge Strips"):
                st.markdown(f"**Dokumen Terpilih:** `{', '.join(docs)}`")
                if strips:
                    st.text(strips)

# Chat Input & Response Handling
prompt = st.chat_input("Ketik pertanyaan akademik di sini...")

if prompt:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt, "meta": None})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Determine effective prodi filter
    if selected_prodi == "Otomatis (Deteksi dari pertanyaan)":
        effective_prodi = detect_prodi(prompt)
    elif selected_prodi == "Semua Program Studi (Umum)":
        effective_prodi = None
    else:
        effective_prodi = selected_prodi

    # Assistant Response Container
    with st.chat_message("assistant"):
        with st.spinner("Mencari referensi & memverifikasi relevansi dokumen..."):
            status, max_conf, cand_ids, refined, fallback_q, gen_prompt, abstain_msg = (
                rag.prepare_crag_context(prompt, top_k=top_k, prodi_filter=effective_prodi)
            )

        badge_class = (
            "status-badge-correct"
            if status == "CORRECT"
            else ("status-badge-ambiguous" if status == "AMBIGUOUS" else "status-badge-incorrect")
        )
        st.markdown(
            f'<span class="{badge_class}">CRAG: {status} (Keyakinan: {max_conf:.2f})</span>',
            unsafe_allow_html=True,
        )

        if abstain_msg:
            st.markdown(abstain_msg)
            full_response = abstain_msg
        else:
            # Stream response directly from Ollama
            def stream_gen():
                for token in stream_with_ollama(gen_prompt, model=selected_model):
                    yield token

            full_response = st.write_stream(stream_gen())

        # Expandable inspection
        with st.expander("🔍 Lihat Dokumen Sitasi & Knowledge Strips"):
            st.markdown(f"**Dokumen Terpilih:** `{', '.join(cand_ids)}`")
            if effective_prodi:
                st.markdown(f"**Filter Jurusan:** `{effective_prodi}`")
            if refined:
                st.text(refined)

    # Save assistant message to session state
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_response,
            "meta": {
                "status": status,
                "conf": max_conf,
                "docs": cand_ids,
                "strips": refined,
            },
        }
    )
