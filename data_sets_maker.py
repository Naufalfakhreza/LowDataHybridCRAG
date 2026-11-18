import os
import re
import json
import random
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from google import genai
from config import GEMINI_API_KEY

@dataclass
class DocumentChunk:
    doc_id: str
    text: str
    metadata: Dict[str, str]


@dataclass
class RAFTEvalSample:
    question_id: str
    question: str
    oracle_doc_id: str
    oracle_context: str
    distractor_doc_ids: List[str]
    cot_Verify_answer: str
    prodi_filter: str


class BPADatasetMaker:
    """
    1. Chunks hierarchical Markdown while keeping tables and course blocks intact.
    2. Generates a RAFT-style synthetic evaluation dataset with hard-negative distractors.
    """

    def __init__(self, md_path: str, gemini_api_key: Optional[str] = None):
        self.md_path = md_path
        self.client = genai.Client(api_key=gemini_api_key) if gemini_api_key else None

    def build_hierarchical_chunks(
        self, output_chunks_jsonl: str, max_chars: int = 1600
    ) -> List[DocumentChunk]:
        with open(self.md_path, "r", encoding="utf-8") as f:
            md_text = f.read()

        # Split along Markdown headers (#, ##, ###)
        raw_sections = re.split(r"(?=\n#{1,3}\s+)", md_text)
        chunks: List[DocumentChunk] = []

        current_bagian = "Pra-Kata & SK Rektor"
        current_prodi = "Universitas Negeri Jakarta"
        current_sub = "Umum"
        counter = 1

        for sec in raw_sections:
            sec = sec.strip()
            if not sec:
                continue

            first_line = sec.splitlines()[0].strip()
            if first_line.startswith("# "):
                current_bagian = first_line.lstrip("# ").strip()
                current_prodi = "Umum"
                current_sub = "Pengantar"
            elif first_line.startswith("## "):
                current_prodi = first_line.lstrip("# ").strip()
                current_sub = "Ringkasan"
            elif first_line.startswith("### "):
                current_sub = first_line.lstrip("# ").strip()

            breadcrumb = f"[{current_bagian} > {current_prodi} > {current_sub}]"
            meta = {
                "bagian": current_bagian,
                "prodi": current_prodi,
                "subsection": current_sub,
            }

            # Keep section atomic if under max_chars
            if len(sec) <= max_chars:
                chunks.append(
                    DocumentChunk(
                        doc_id=f"BPA-{counter:04d}",
                        text=f"{breadcrumb}\n{sec}",
                        metadata=meta,
                    )
                )
                counter += 1
            else:
                # Split long sections (e.g. G. Deskripsi Mata Kuliah) by double newline
                # while keeping Markdown tables intact and prepending breadcrumb to every sub-chunk
                blocks = re.split(r"\n\n+", sec)
                buffer = ""
                for blk in blocks:
                    if len(buffer) + len(blk) > max_chars and buffer:
                        chunks.append(
                            DocumentChunk(
                                doc_id=f"BPA-{counter:04d}",
                                text=f"{breadcrumb}\n{buffer.strip()}",
                                metadata=meta,
                            )
                        )
                        counter += 1
                        buffer = blk + "\n\n"
                    else:
                        buffer += blk + "\n\n"

                if buffer.strip():
                    chunks.append(
                        DocumentChunk(
                            doc_id=f"BPA-{counter:04d}",
                            text=f"{breadcrumb}\n{buffer.strip()}",
                            metadata=meta,
                        )
                    )
                    counter += 1

        with open(output_chunks_jsonl, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

        print(f"[data_sets_maker] Saved {len(chunks)} hierarchical chunks to: {output_chunks_jsonl}")
        return chunks

    def _select_hard_distractors(
        self, oracle: DocumentChunk, all_chunks: List[DocumentChunk], k: int = 2
    ) -> List[DocumentChunk]:
        """
        Selects hard-negative distractors: chunks with the SAME subsection name
        (e.g., 'F. Kurikulum' or 'G. Deskripsi Mata Kuliah') but from DIFFERENT study programs.
        """
        same_sub_diff_prodi = [
            c
            for c in all_chunks
            if c.doc_id != oracle.doc_id
            and c.metadata.get("subsection") == oracle.metadata.get("subsection")
            and c.metadata.get("prodi") != oracle.metadata.get("prodi")
        ]
        if len(same_sub_diff_prodi) >= k:
            return random.sample(same_sub_diff_prodi, k)

        fallback_pool = [c for c in all_chunks if c.doc_id != oracle.doc_id]
        return random.sample(fallback_pool, min(k, len(fallback_pool)))

    def generate_raft_eval_dataset(
        self,
        chunks: List[DocumentChunk],
        output_eval_jsonl: str,
        num_samples: int = 15,
    ) -> List[RAFTEvalSample]:
        """
        Generates synthetic QA + Chain-of-Thought evaluation samples using RAFT principles.
        """
        eligible = [c for c in chunks if len(c.text) > 250]
        sampled_oracles = random.sample(eligible, min(num_samples, len(eligible)))
        eval_samples: List[RAFTEvalSample] = []

        for idx, oracle in enumerate(sampled_oracles, start=1):
            distractors = self._select_hard_distractors(oracle, chunks, k=2)
            prodi = oracle.metadata.get("prodi", "Umum")
            sub = oracle.metadata.get("subsection", "Umum")

            if self.client:
                prompt = (
                    "Anda adalah pembuat dataset evaluasi akademik Universitas Negeri Jakarta (UNJ).\n"
                    "Berdasarkan potongan dokumen resmi (Oracle Chunk) berikut, buatlah 1 pertanyaan spesifik "
                    "yang menyebutkan nama Program Studi atau Bagian terkait, serta jawaban faktual lengkap "
                    "dengan penalaran singkat yang mengutip kode dokumen.\n"
                    "Format output JSON: {\"question\": \"...\", \"cot_answer\": \"...\"}\n\n"
                    f"Oracle ID: {oracle.doc_id}\n"
                    f"Oracle Chunk:\n{oracle.text}"
                )
                try:
                    resp = self.client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt,
                        config={"response_mime_type": "application/json"},
                    )
                    parsed = json.loads(resp.text)
                    question = parsed.get("question", "")
                    cot_answer = parsed.get("cot_answer", "")
                except Exception:
                    question = f"Apa ketentuan atau informasi yang diatur pada bagian {sub} untuk {prodi}?"
                    cot_answer = f"Berdasarkan [{oracle.doc_id}], informasi pada {prodi} ({sub}) mencakup: {oracle.text[:220]}..."
            else:
                question = f"Apa rincian ketentuan akademik pada bagian {sub} di {prodi}?"
                cot_answer = (
                    f"Berdasarkan dokumen [{oracle.doc_id}] pada jalur {prodi} > {sub}, "
                    f"ketentuan yang berlaku adalah: {oracle.text[:250]}..."
                )

            eval_samples.append(
                RAFTEvalSample(
                    question_id=f"Q-{idx:03d}",
                    question=question,
                    oracle_doc_id=oracle.doc_id,
                    oracle_context=oracle.text,
                    distractor_doc_ids=[d.doc_id for d in distractors],
                    cot_Verify_answer=cot_answer,
                    prodi_filter=prodi,
                )
            )

        with open(output_eval_jsonl, "w", encoding="utf-8") as f:
            for s in eval_samples:
                f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")

        print(f"[data_sets_maker] Saved {len(eval_samples)} RAFT evaluation pairs to: {output_eval_jsonl}")
        return eval_samples


if __name__ == "__main__":
    # Create a small demo Markdown file if bpa_fmipa_unj_2026.md doesn't exist yet
    md_file = "bpa_fmipa_unj_2026.md"
    if not os.path.exists(md_file):
        sample_md = (
            "# Bagian 2: Penyelenggaraan Pendidikan\n"
            "## Umum\n"
            "### D. Pengisian Jumlah SKS\n"
            "**[Context Path: Bagian 2 > Umum > D. Pengisian Jumlah SKS | Page 32]**\n"
            "Semester 1 dan 2 paket maksimal 20 SKS. Semester 3 dst berdasarkan IPS: "
            "IPS >= 3,30 maksimal 24 SKS; IPS 2,75 - 3,29 maksimal 22 SKS; IPS 2,00 - 2,74 maksimal 20 SKS; "
            "IPS < 2,00 maksimal 16 SKS. Mahasiswa aktif kembali setelah mangkir maksimal 16 SKS.\n\n"
            "# Bagian 3: Panduan Operasional Akademik FMIPA\n"
            "## Program Studi S-1 Pendidikan Matematika\n"
            "### F. Kurikulum\n"
            "**[Context Path: Bagian 3 > Program Studi S-1 Pendidikan Matematika > F. Kurikulum | Page 57]**\n"
            "| No | Kode MK | Nama MK | SKS | Semester Pengambilan |\n"
            "| --- | --- | --- | --- | --- |\n"
            "| 1 | 0005-111-2 | Pancasila | 2 | Semester 1 |\n"
            "| 6 | 0005-322-2 | Data Raya dan Pemrograman | 2 | Semester 2 |\n"
            "| 38 | 3005-402-4 | Skripsi | 4 | Semester 8 |\n\n"
            "## Program Studi S-1 Pendidikan Fisika\n"
            "### G. Deskripsi Mata Kuliah\n"
            "**[Context Path: Bagian 3 > Program Studi S-1 Pendidikan Fisika > G. Deskripsi Mata Kuliah | Page 93]**\n"
            "Skripsi (6 SKS): Mata kuliah ini melakukan penelitian ilmiah sesuai proposal yang telah diajukan "
            "dan dipresentasikan dalam sidang skripsi.\n"
        )
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(sample_md)

    maker = BPADatasetMaker(
        md_path="bpa_fmipa_unj_2026.md",
        gemini_api_key=GEMINI_API_KEY,
    )
    chunks = maker.build_hierarchical_chunks(output_chunks_jsonl="bpa_corpus_chunks.jsonl")
    maker.generate_raft_eval_dataset(
        chunks=chunks,
        output_eval_jsonl="bpa_eval_raft_dataset.jsonl",
        num_samples=5,
    )