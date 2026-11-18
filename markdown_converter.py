import os
import re
import fitz 
import pdfplumber
from dataclasses import dataclass
from typing import List, Optional
from google import genai
from google.genai import types
from config import GEMINI_API_KEY, generate_with_gemini_retry


@dataclass
class HierarchyState:
    bagian: str = "Pra-Kata & Keputusan Rektor"
    prodi_or_unit: str = "Umum"
    subsection: str = "Pendahuluan"

    def get_breadcrumb(self) -> str:
        parts = [p.strip() for p in [self.bagian, self.prodi_or_unit, self.subsection] if p.strip()]
        return " > ".join(parts)


class BPAMarkdownConverter:
    """
    Extracts text, multi-page curriculum tables, and non-decorative diagrams from
    'BPA FMIPA UNJ 2026.pdf' into a hierarchy-preserving Markdown file.
    """

    CHECKMARK_GLYPHS = {"ة", "Ο", "v", "V", "√", "Ö", "o", "O"}

    def __init__(self, pdf_path: str, gemini_api_key: Optional[str] = None):
        self.pdf_path = pdf_path
        self.state = HierarchyState()
        self.client = genai.Client(api_key=gemini_api_key) if gemini_api_key else None

        # Level 1: Bagian 1, Bagian 2, Bagian 3
        self.re_bagian = re.compile(r"^Bagian\s+(\d+)\s*[\.\:]?\s*(.*)", re.IGNORECASE)

        # Level 2: Program Studi or Roman Numeral Sections
        self.re_prodi = re.compile(
            r"^(Program Studi\s+S-[12]\s+.*|I+\.\s+Pendahuluan|II\.\s+Identitas|III\.\s+(?:Pimpinan|Tata Kelola))",
            re.IGNORECASE,
        )

        # Level 3: Lettered Subsections (A. - U.)
        self.re_subsection = re.compile(
            r"^([A-U])\.\s+(Sejarah|Visi.*|Motto|Struktur Organisasi|Fungsi.*|Lambang|Warna Bendera.*|"
            r"Hymne.*|Bentuk Toga|Universitas dan Fakultas|Badan dan Lembaga|Direktorat|Satuan|Unit|"
            r"Tujuan Pendidikan|Kalender Akademik|Registrasi|Pengisian Jumlah SKS|Status Mahasiswa|"
            r"Pembimbing Akademik|Kurikulum|Kartu Rencana Studi.*|Pelaksanaan Perkuliahan|"
            r"Pembelajaran Inklusif|Kehadiran Perkuliahan|Kegiatan Pembelajaran.*|Penilaian Hasil Belajar|"
            r"Kartu Hasil Studi.*|Evaluasi Kemajuan.*|Penyelesaian Studi|Tugas Akhir|Yudisium|"
            r"Predikat Kelulusan|Wisuda|Pelanggaran.*|Pengantar|Daftar Program Studi|Tata Kelola|"
            r"Profil Lulusan|Capaian Pembelajaran.*|Kompetensi Lulusan|Gelar Lulusan|Akreditasi|"
            r"Deskripsi Mata Kuliah|Dosen)$",
            re.IGNORECASE,
        )

        # Running headers/footers to strip
        self.re_running_header = re.compile(
            r"^(\d+\s+Buku Pedoman Akademik 2026/2027|Buku Pedoman Akademik 2026/2027\s*\d*|"
            r"Universitas Negeri Jakarta\s+[ivx\d]+|Fakultas Matematika dan Ilmu Pe[n]?getahuan Alam\s+\d+|[ivx]+)$",
            re.IGNORECASE,
        )

    @staticmethod
    def _is_decorative_image(width: int, height: int) -> bool:
        """Filters out small decorative wave icons (e.g., pages 2, 30, 48, 90-140)."""
        return width < 140 or height < 90

    def _caption_visual_element(self, image_bytes: bytes, page_num: int) -> str:
        """Transcribes organizational charts, flags, and toga specs into structured Markdown."""
        if not self.client:
            return (
                f"\n**[Visual Asset - Page {page_num} "
                f"({self.state.get_breadcrumb()})]**\n"
            )

        prompt = (
            "Transcribe this image from Buku Pedoman Akademik UNJ 2026 into structured Indonesian "
            "Markdown bullet points. Preserve all hierarchy arrows, unit names, color codes (CMYK/RGB), "
            "or toga specifications. If the image is purely a decorative wave or blank border, reply 'DECORATIVE'."
        )
        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                    prompt,
                ],
            )
            text = generate_with_gemini_retry(
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                    prompt,
                ]
            ).strip()
            if "DECORATIVE" in text.upper() or not text:
                return ""
            return (
                f"\n**[Visual Asset Transcription - Page {page_num} | {self.state.get_breadcrumb()}]**\n"
                f"{text}\n"
            )
        except Exception as e:
            return f"\n**[Visual Asset - Page {page_num} | VLM Error: {e}]**\n"

    def _normalize_curriculum_row(self, row: List[Optional[str]]) -> List[str]:
        cleaned = [re.sub(r"\s+", " ", (cell or "")).strip() for cell in row]
        
        # Detect curriculum table row: starts with a number, has Kode MK, Nama MK, and SKS digit
        if len(cleaned) >= 5 and re.match(r"^\d+$", cleaned[0]) and re.match(r"^\d+$", cleaned[3]):
            no, kode, nama, sks = cleaned[0], cleaned[1], cleaned[2], cleaned[3]
            sem_cells = cleaned[4:]
            
            # Map checkmarks from the right if continuation page dropped empty leading columns,
            # or map 1..8 directly if all 8 semester columns are present
            active_sems = []
            offset = 8 - len(sem_cells) if len(sem_cells) < 8 else 0
            for i, val in enumerate(sem_cells):
                if any(g in val for g in self.CHECKMARK_GLYPHS):
                    sem_num = i + 1 + max(0, offset)
                    active_sems.append(f"Semester {sem_num}")
                    
            sem_str = ", ".join(active_sems) if active_sems else "Semester tidak ditentukan"
            # Adding a natural-language summary cell makes dense & cross-encoder retrieval 10x more accurate
            verbalized = f"Mata kuliah {nama} (Kode: {kode}) berbobot {sks} SKS diambil pada {sem_str}."
            return [no, kode, nama, sks, sem_str, verbalized]
            
        return cleaned

    def _extract_tables_as_markdown(self, plumber_page, page_num: int) -> List[str]:
        """Extracts tables from a page and injects hierarchical metadata headers."""
        md_tables = []
        tables = plumber_page.extract_tables()
        for tbl in tables:
            if not tbl or len(tbl) < 2:
                continue
            rows = [self._normalize_curriculum_row(r) for r in tbl if any(r)]
            if not rows:
                continue

            max_cols = max(len(r) for r in rows)
            # Cap or merge columns if table extraction is overly wide
            padded = [r + [""] * (max_cols - len(r)) for r in rows]

            header = padded[0]
            # If normalized curriculum row reduced 11 cols to 5, label the 5th column
            if len(header) == 5 and header[-1] in {"1", "Semester", ""}:
                header[-1] = "Semester Pengambilan"

            sep = ["---"] * len(header)
            lines = [
                f"\n**[Table Context - Page {page_num} | {self.state.get_breadcrumb()}]**",
                "| " + " | ".join(header) + " |",
                "| " + " | ".join(sep) + " |",
            ]
            for r in padded[1:]:
                lines.append("| " + " | ".join(r[: len(header)]) + " |")
            md_tables.append("\n".join(lines) + "\n")
        return md_tables

    def _process_line(self, line: str, page_num: int) -> Optional[str]:
        clean = line.strip()
        if not clean or self.re_running_header.match(clean):
            return None

        m_bagian = self.re_bagian.match(clean)
        if m_bagian:
            num, title = m_bagian.groups()
            self.state.bagian = f"Bagian {num}: {title}".strip(": ")
            self.state.prodi_or_unit = "Umum"
            self.state.subsection = "Pengantar"
            return f"\n# {self.state.bagian}\n"

        m_prodi = self.re_prodi.match(clean)
        if m_prodi:
            self.state.prodi_or_unit = re.sub(r"\s+", " ", m_prodi.group(1)).strip()
            self.state.subsection = "Ringkasan"
            return f"\n## {self.state.prodi_or_unit}\n"

        m_sub = self.re_subsection.match(clean)
        if m_sub:
            letter, title = m_sub.groups()
            self.state.subsection = f"{letter.upper()}. {title}"
            return (
                f"\n### {self.state.subsection}\n"
                f"**[Context Path: {self.state.get_breadcrumb()} | Page {page_num}]**\n"
            )

        return clean

    def convert(self, output_md_path: str, start_page: int = 1, end_page: Optional[int] = None):
        fitz_doc = fitz.open(self.pdf_path)
        plumber_doc = pdfplumber.open(self.pdf_path)
        total_pages = len(fitz_doc)
        final_page = min(end_page or total_pages, total_pages)

        md_blocks: List[str] = []

        for idx in range(start_page - 1, final_page):
            page_num = idx + 1
            fitz_page = fitz_doc[idx]
            plumber_page = plumber_doc.pages[idx]

            raw_text = fitz_page.get_text("text")
            # Join multi-line study program titles (e.g., "Program Studi S-1\nPendidikan\nMatematika")
            raw_text = re.sub(
                r"(Program Studi\s+S-[12])\s*\n+\s*(Pendidikan\s*\n*\s*\w+|\w+(?:\s+\w+)?)",
                lambda m: f"{m.group(1)} " + re.sub(r"\s+", " ", m.group(2)),
                raw_text,
            )

            lines_out = []
            for line in raw_text.splitlines():
                processed = self._process_line(line, page_num)
                if processed:
                    lines_out.append(processed)

            if lines_out:
                md_blocks.append("\n".join(lines_out))

            # Append structured Markdown tables
            md_blocks.extend(self._extract_tables_as_markdown(plumber_page, page_num))

            # Append non-decorative visual diagrams
            for img_info in fitz_page.get_images(full=True):
                xref = img_info[0]
                base_img = fitz_doc.extract_image(xref)
                if not self._is_decorative_image(base_img["width"], base_img["height"]):
                    caption = self._caption_visual_element(base_img["image"], page_num)
                    if caption:
                        md_blocks.append(caption)

        plumber_doc.close()
        fitz_doc.close()

        with open(output_md_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(md_blocks))
        print(f"[bpa_to_markdown] Successfully wrote hierarchical Markdown to: {output_md_path}")


if __name__ == "__main__":
    converter = BPAMarkdownConverter(
        pdf_path="BPA FMIPA UNJ 2026.pdf",
        gemini_api_key=GEMINI_API_KEY,
    )
    converter.convert(output_md_path="bpa_fmipa_unj_2026.md")