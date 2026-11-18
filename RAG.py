# File: RAG.py
import os
import re
import json
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict, Tuple, Callable, Optional
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder, util

from config import generate_with_ollama, stream_with_ollama, OLLAMA_MODEL

PRODI_PATTERNS = [
    (r"\b(s-?1\s+)?pendidikan\s+matematika\b", "Pendidikan Matematika"),
    (r"\b(s-?2\s+)?pendidikan\s+matematika\b", "Program Studi S-2 Pendidikan Matematika"),
    (r"\b(s-?1\s+)?pendidikan\s+biologi\b", "Pendidikan Biologi"),
    (r"\b(s-?2\s+)?pendidikan\s+biologi\b", "Program Studi S-2 Pendidikan Biologi"),
    (r"\b(s-?1\s+)?pendidikan\s+fisika\b", "Pendidikan Fisika"),
    (r"\b(s-?2\s+)?pendidikan\s+fisika\b", "Program Studi S-2 Pendidikan Fisika"),
    (r"\b(s-?1\s+)?pendidikan\s+kimia\b", "Pendidikan Kimia"),
    (r"\b(s-?2\s+)?pendidikan\s+kimia\b", "Program Studi S-2 Pendidikan Kimia"),
    (r"\b(s-?1\s+)?ilmu\s+komputer\b|\bilkom\b", "Ilmu Komputer"),
    (r"\b(s-?1\s+)?statistika\b", "Statistika"),
    (r"\b(s-?1\s+)?matematika\b", "Matematika"),
    (r"\b(s-?1\s+)?biologi\b", "Biologi"),
    (r"\b(s-?1\s+)?fisika\b", "Fisika"),
    (r"\b(s-?1\s+)?kimia\b", "Kimia"),
]


def detect_prodi(query: str) -> Optional[str]:
    """Auto-detects study program mentioned in user query."""
    q_low = query.lower()
    for pattern, prodi_name in PRODI_PATTERNS:
        if re.search(pattern, q_low):
            return prodi_name
    return None


@dataclass
class DocumentChunk:
    doc_id: str
    text: str
    metadata: Dict[str, str]


@dataclass
class CRAGResponse:
    query: str
    status: str  # "CORRECT", "AMBIGUOUS", or "INCORRECT"
    max_confidence: float
    retrieved_doc_ids: List[str]
    refined_context: str
    fallback_search_query: Optional[str]
    answer: str


class BPAHybridCRAG:
    """
    Local Ollama Hybrid CRAG Engine for BPA FMIPA UNJ 2026:
    - Lexical Okapi BM25 on raw query + hierarchical breadcrumbs
    - HyDE (Hypothetical Document Embeddings) generated via local Ollama
    - Reciprocal Rank Fusion (RRF)
    - CRAG Cross-Encoder relevance evaluator + table-aware knowledge strip filtering
    """

    def __init__(
        self,
        chunks_jsonl_path: str,
        llm_callable: Callable[[str], str] = generate_with_ollama,
        bi_encoder_name: str = "intfloat/multilingual-e5-small",
        cross_encoder_name: str = "BAAI/bge-reranker-v2-m3",
        rrf_k: int = 60,
        theta_high: float = 0.55,
        theta_low: float = 0.20,
    ):
        self.corpus = self._load_chunks(chunks_jsonl_path)
        self.llm = llm_callable
        self.rrf_k = rrf_k
        self.theta_high = theta_high
        self.theta_low = theta_low

        print(f"[RAG Init] Loaded {len(self.corpus)} chunks. Building BM25 & Dense indices...")
        self.tokenized_corpus = [self._tokenize(c.text) for c in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        self.bi_encoder = SentenceTransformer(bi_encoder_name)
        passage_texts = [f"passage: {c.text}" for c in self.corpus]
        self.doc_embeddings = self.bi_encoder.encode(
            passage_texts,
            normalize_embeddings=True,
            convert_to_tensor=True,
            show_progress_bar=True,
        )

        print(f"[RAG Init] Loading Cross-Encoder ({cross_encoder_name})...")
        self.evaluator = CrossEncoder(cross_encoder_name)

    @staticmethod
    def _load_chunks(path: str) -> List[DocumentChunk]:
        chunks = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(DocumentChunk(**json.loads(line)))
        return chunks

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9\-]+", text.lower())

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def generate_hyde_passage(self, query: str) -> str:
        """Uses local Ollama to generate a zero-shot hypothetical BPA passage."""
        prompt = (
            "Tuliskan tepat satu paragraf singkat (maksimal 3 kalimat) kutipan Buku Pedoman Akademik "
            "Universitas Negeri Jakarta (UNJ) yang menjawab pertanyaan berikut secara faktual "
            "(sebutkan istilah SKS, Semester, IPS, atau Kode MK yang relevan).\n"
            f"Pertanyaan: {query}\n"
            "Paragraf Hipotetis:"
        )
        return self.llm(prompt)

    def retrieve_hybrid_rrf(
        self, query: str, top_k: int = 5, prodi_filter: Optional[str] = None
    ) -> List[Tuple[int, float]]:
        valid_indices = [
            i
            for i, c in enumerate(self.corpus)
            if not prodi_filter
            or prodi_filter.lower() in c.metadata.get("prodi", "").lower()
            or c.metadata.get("prodi", "") == "Umum"
        ]
        if not valid_indices:
            valid_indices = list(range(len(self.corpus)))

        # 1. Lexical BM25 on raw user query
        bm25_all = self.bm25.get_scores(self._tokenize(query))
        bm25_ranked = sorted(valid_indices, key=lambda idx: bm25_all[idx], reverse=True)

        # 2. Dense similarity on Ollama-generated HyDE passage
        hyde_doc = self.generate_hyde_passage(query)
        hyde_emb = self.bi_encoder.encode(
            f"query: { query } { hyde_doc }",
            normalize_embeddings=True,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        dense_all = util.dot_score(hyde_emb, self.doc_embeddings)[0].cpu().numpy()
        dense_ranked = sorted(valid_indices, key=lambda idx: dense_all[idx], reverse=True)

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores: Counter = Counter()
        for rank, idx in enumerate(bm25_ranked, start=1):
            rrf_scores[idx] += 1.0 / (self.rrf_k + rank)
        for rank, idx in enumerate(dense_ranked, start=1):
            rrf_scores[idx] += 1.0 / (self.rrf_k + rank)

        return rrf_scores.most_common(top_k)

    def refine_knowledge_strips(
        self, query: str, candidate_indices: List[int], strip_threshold: float = 0.20, max_strips: int = 15
    ) -> str:
        """
        Splits candidate chunks into sentences/table rows and filters noise via Cross-Encoder,
        while preserving Markdown table rows that lexically match the query.
        """
        query_tokens = set(self._tokenize(query))
        strips: List[Tuple[str, str, str]] = []

        for idx in candidate_indices:
            chunk = self.corpus[idx]
            lines = chunk.text.splitlines()
            breadcrumb = lines[0] if lines else f"[{chunk.doc_id}]"
            body = "\n".join(lines[1:]) if len(lines) > 1 else chunk.text

            for line in body.splitlines():
                line = line.strip()
                if not line or line.startswith("| ---") or line.startswith("|---"):
                    continue
                if line.startswith("|"):
                    # Preserve table row intact
                    strips.append((chunk.doc_id, breadcrumb, line))
                else:
                    for s in re.split(r"(?<=[.!?])\s+", line):
                        s = s.strip()
                        if len(s) > 10:
                            strips.append((chunk.doc_id, breadcrumb, s))

        if not strips:
            return ""

        # Score segment relevance directly without breadcrumb inflation
        pairs = [[query, seg] for _, _, seg in strips]
        logits = np.array(self.evaluator.predict(pairs))
        probs = self._sigmoid(logits)

        scored = []
        for (doc_id, bc, seg), prob in zip(strips, probs):
            seg_tokens = set(self._tokenize(seg))
            is_matching_table_row = seg.startswith("|") and len(query_tokens & (seg_tokens | set(self._tokenize(bc)))) >= 2
            boosted_prob = prob + (0.25 if is_matching_table_row else 0.0)
            if prob >= strip_threshold or is_matching_table_row:
                scored.append((boosted_prob, prob, doc_id, bc, seg))

        # Sort descending by relevance score to prioritize most critical facts and prevent context flooding
        scored.sort(key=lambda x: x[0], reverse=True)
        kept = [
            f"[{doc_id} | {bc} | conf={prob:.2f}] {seg}"
            for _, prob, doc_id, bc, seg in scored[:max_strips]
        ]

        return "\n".join(kept)

    def prepare_crag_context(
        self, user_query: str, top_k: int = 10, prodi_filter: Optional[str] = None
    ) -> Tuple[str, float, List[str], str, Optional[str], Optional[str], Optional[str]]:
        """
        Executes CRAG pipeline up to prompt creation.
        Returns: (status, max_conf, cand_ids, refined_strips, fallback_q, gen_prompt, abstain_msg)
        """
        hits = self.retrieve_hybrid_rrf(user_query, top_k=top_k, prodi_filter=prodi_filter)
        cand_indices = [idx for idx, _ in hits]
        cand_ids = [self.corpus[idx].doc_id for idx in cand_indices]

        eval_pairs = [[user_query, self.corpus[idx].text] for idx in cand_indices]
        logits = np.array(self.evaluator.predict(eval_pairs))
        confidences = self._sigmoid(logits)
        max_conf = float(np.max(confidences)) if len(confidences) > 0 else 0.0

        valid_indices = [
            idx for idx, c in zip(cand_indices, confidences) if c >= self.theta_low
        ]

        if max_conf >= self.theta_high:
            status = "CORRECT"
            refined = self.refine_knowledge_strips(user_query, valid_indices)
            fallback_q = None
        elif max_conf >= self.theta_low:
            status = "AMBIGUOUS"
            refined = self.refine_knowledge_strips(user_query, valid_indices)
            fallback_q = self.llm(
                f"Tuliskan hanya 4-6 kata kunci pencarian tanpa penjelasan untuk: {user_query}"
            )
        else:
            status = "INCORRECT"
            refined = ""
            fallback_q = self.llm(
                f"Tuliskan hanya 4-6 kata kunci pencarian tanpa penjelasan untuk: {user_query}"
            )

        if status == "INCORRECT":
            abstain_msg = (
                f"[ABSTAIN - CRAG INCORRECT] Kepercayaan dokumen lokal rendah "
                f"({max_conf:.2f} < {self.theta_low}). Kata kunci fallback: '{fallback_q}'"
            )
            gen_prompt = None
        else:
            abstain_msg = None
            gen_prompt = (
                "Anda adalah asisten akademik Fakultas MIPA Universitas Negeri Jakarta (UNJ).\n"
                "Jawablah pertanyaan berikut secara akurat HANYA berdasarkan fakta pada Knowledge Strips di bawah ini.\n"
                "Wajib cantumkan sitasi kode dokumen [BPA-XXXX] di akhir setiap klaim.\n\n"
                f"Knowledge Strips:\n{refined}\n\n"
                f"Pertanyaan: {user_query}\n"
                "Jawaban:"
            )

        return status, max_conf, cand_ids, refined, fallback_q, gen_prompt, abstain_msg

    def query(
        self, user_query: str, top_k: int = 10, prodi_filter: Optional[str] = None
    ) -> CRAGResponse:
        status, max_conf, cand_ids, refined, fallback_q, gen_prompt, abstain_msg = (
            self.prepare_crag_context(user_query, top_k=top_k, prodi_filter=prodi_filter)
        )
        answer = abstain_msg if abstain_msg else self.llm(gen_prompt)
        return CRAGResponse(
            query=user_query,
            status=status,
            max_confidence=max_conf,
            retrieved_doc_ids=cand_ids,
            refined_context=refined,
            fallback_search_query=fallback_q,
            answer=answer,
        )

    def query_stream(
        self, user_query: str, top_k: int = 10, prodi_filter: Optional[str] = None
    ):
        """
        Runs CRAG retrieval and yields tokens as they are generated by Ollama.
        Returns: (status, max_conf, cand_ids, refined_context, fallback_query, token_generator)
        """
        status, max_conf, cand_ids, refined, fallback_q, gen_prompt, abstain_msg = (
            self.prepare_crag_context(user_query, top_k=top_k, prodi_filter=prodi_filter)
        )

        def token_stream():
            if abstain_msg:
                yield abstain_msg
            else:
                for token in stream_with_ollama(gen_prompt):
                    yield token

        return status, max_conf, cand_ids, refined, fallback_q, token_stream()

    def benchmark_percentiles(self, eval_jsonl_path: str, top_k: int = 5) -> Dict[str, float]:
        samples = []
        with open(eval_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))

        max_confs = []
        hits_at_k = 0

        for s in samples:
            res = self.query(s["question"], top_k=top_k)
            max_confs.append(res.max_confidence)
            if s["oracle_doc_id"] in res.retrieved_doc_ids:
                hits_at_k += 1

        arr = np.array(max_confs)
        stats = {
            "total_queries": len(samples),
            f"recall@{top_k}": hits_at_k / max(1, len(samples)),
            "conf_min": float(np.min(arr)),
            "conf_p25": float(np.percentile(arr, 25)),
            "conf_median_p50": float(np.percentile(arr, 50)),
            "conf_p75": float(np.percentile(arr, 75)),
            "conf_p95": float(np.percentile(arr, 95)),
            "conf_max": float(np.max(arr)),
        }
        print(f"\n=== CRAG Multi-Percentile Benchmark (Ollama: {OLLAMA_MODEL}) ===")
        for k, v in stats.items():
            print(f"{k:<18}: {v:.4f}" if isinstance(v, float) else f"{k:<18}: {v}")
        return stats


def interactive_cli(rag: BPAHybridCRAG):
    """Interactive real-time CLI chat loop."""
    print("\n" + "=" * 65)
    print("🎓  Selamat Datang di Asisten Akademik FMIPA UNJ 2026 (Hybrid CRAG)")
    print("=" * 65)
    print("Ketik pertanyaan akademik seputar kurikulum, SKS, dosen, atau aturan.")
    print("Perintah khusus:")
    print("  /prodi <nama>  -> Kunci filter program studi (misal: /prodi matematika)")
    print("  /prodi auto    -> Aktifkan deteksi prodi otomatis dari pertanyaan (default)")
    print("  /prodi clear   -> Hapus filter prodi (cari di seluruh FMIPA)")
    print("  /k <angka>     -> Ubah retrieval depth top_k (default: 10)")
    print("  keluar / exit  -> Berhenti dari sesi chat")
    print("=" * 65)

    current_prodi_filter: Optional[str] = "auto"
    current_top_k: int = 10

    while True:
        try:
            user_input = input("\n👤 Anda: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSampai jumpa!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "keluar"):
            print("Sampai jumpa!")
            break

        if user_input.startswith("/prodi"):
            parts = user_input.split(maxsplit=1)
            if len(parts) == 1 or parts[1].lower() in ("clear", "none", "semua"):
                current_prodi_filter = None
                print(">> Filter prodi dinonaktifkan (mencakup semua jurusan).")
            elif parts[1].lower() == "auto":
                current_prodi_filter = "auto"
                print(">> Filter prodi diatur ke: AUTO (mendeteksi dari pertanyaan).")
            else:
                current_prodi_filter = parts[1]
                print(f">> Filter prodi dikunci ke: '{current_prodi_filter}'")
            continue

        if user_input.startswith("/k"):
            parts = user_input.split()
            if len(parts) > 1 and parts[1].isdigit():
                current_top_k = int(parts[1])
                print(f">> Retrieval depth diatur ke: top_k={current_top_k}")
            continue

        # Determine effective study program filter
        if current_prodi_filter == "auto":
            effective_filter = detect_prodi(user_input)
            if effective_filter:
                print(f"   [🔍 Auto-detected Prodi: {effective_filter}]")
        else:
            effective_filter = current_prodi_filter

        print(f"   [⏳ Mencari di {len(rag.corpus)} dokumen (top_k={current_top_k})...]")
        status, max_conf, cand_ids, refined, fallback_q, token_gen = rag.query_stream(
            user_input, top_k=current_top_k, prodi_filter=effective_filter
        )

        badge = "🟢 CORRECT" if status == "CORRECT" else ("🟡 AMBIGUOUS" if status == "AMBIGUOUS" else "🔴 INCORRECT")
        print(f"   [{badge} | Max Confidence: {max_conf:.2f} | Docs: {cand_ids[:4]}]")
        print(f"\n🤖 Asisten: ", end="", flush=True)

        for token in token_gen:
            print(token, end="", flush=True)
        print()


if __name__ == "__main__":
    import sys
    print(f"Initializing RAG with local Ollama model: {OLLAMA_MODEL}")
    rag = BPAHybridCRAG(
        chunks_jsonl_path="bpa_corpus_chunks.jsonl",
        llm_callable=generate_with_ollama,
    )

    if "--test" in sys.argv:
        q1 = "Berapa jumlah SKS mata kuliah Skripsi di Program Studi S-1 Pendidikan Matematika dan pada semester berapa diambil?"
        res1 = rag.query(q1, top_k=10, prodi_filter="Pendidikan Matematika")
        print(f"\nQuery: {res1.query}")
        print(f"Status: {res1.status} | Max Confidence: {res1.max_confidence:.3f} | Docs: {res1.retrieved_doc_ids}")
        print(f"\nRefined Strips:\n{res1.refined_context}")
        print(f"\nFinal Answer (Ollama - {OLLAMA_MODEL}):\n{res1.answer}")
    else:
        interactive_cli(rag)
