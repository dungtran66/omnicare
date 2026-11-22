"""Tầng Hybrid RAG: kết hợp 3 tầng truy hồi (BM25 Lexical + TF-IDF Vector Cosine + Cross-Encoder Reranker).

Hỗ trợ đọc và tự động cắt chunk từ các định dạng: Markdown (.md), Text (.txt), CSV (.csv), Word (.docx), Excel (.xlsx), PDF (.pdf).
"""
from __future__ import annotations

import base64
import csv
import io
import math
import re
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from . import config

_STOPWORDS = {
    "la", "va", "cua", "co", "cho", "toi", "minh", "ban", "the", "nao", "khong", "duoc", "thi", "nhu",
    "voi", "mot", "cac", "nhung", "da", "se", "bi", "o", "tai", "ve", "nay", "do", "gi", "a", "oi", "em", "anh", "chi",
    "lam", "sao", "shop", "bao", "lau", "muon", "can", "hoi", "nhieu", "nghin", "k", "khi",
}

# Từ đồng nghĩa dùng chung cho mọi shop. Từ riêng ngành (vd thời trang: "chật" -> đổi size)
# khai báo ở `synonyms` trong shops/<id>/shop.json và được gộp vào khi shop đó đang bật.
_BASE_SYNONYMS: dict[str, list[str]] = {
    "cuoc": ["phi", "van_chuyen"],
    "ship": ["van_chuyen", "giao_hang"],
    "freeship": ["mien_phi", "van_chuyen"],
    "hu": ["bao_hanh", "loi"],
    "hong": ["bao_hanh", "loi"],
    "quet": ["thanh_toan", "the"],
    "chuyen_khoan": ["thanh_toan"],
    "tra_hang": ["doi_tra", "hoan_tien"],
    "hoan": ["hoan_tien"],
    "lai_tien": ["hoan_tien"],
    "lay_tai": ["cua_hang", "nhan"],
    "showroom": ["cua_hang"],
    "tich_diem": ["thanh_vien", "diem"],
    "voucher": ["diem", "thanh_vien"],
    "sale": ["giam_gia", "chuong_trinh"],
    "khuyen_mai": ["giam_gia", "chuong_trinh"],
    "giam": ["giam_gia"],
}


def _synonyms() -> dict[str, list[str]]:
    from .shop import active  # import muộn để tránh vòng import

    return {**_BASE_SYNONYMS, **active().synonyms}

# Đoạn dài hơn ngưỡng này sẽ bị cắt nhỏ (tài liệu upload thường chỉ có 1 mục rất dài)
CHUNK_MAX_CHARS = 900
CHUNK_OVERLAP_SENTENCES = 1
MIN_QUERY_COVERAGE = 0.34


def normalize(text: str) -> str:
    text = text.lower().replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def tokenize(text: str, expand_synonyms: bool = False) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", normalize(text))
    words = [t for t in tokens if t not in _STOPWORDS]
    bigrams = [f"{a}_{b}" for a, b in zip(words, words[1:])]
    out = words + bigrams
    if expand_synonyms:
        syn = _synonyms()
        extra: list[str] = []
        for w in words + bigrams:
            if w in syn:
                extra.extend(syn[w])
        out.extend(extra)
    return out


def _char_trigrams(text: str) -> Counter:
    clean = re.sub(r"\s+", " ", normalize(text)).strip()
    if len(clean) < 3:
        return Counter()
    return Counter(clean[i:i + 3] for i in range(len(clean) - 2))


def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    if not common:
        return 0.0
    dot = sum(a[k] * b[k] for k in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def parse_document_to_markdown(filename: str, content: str = "", base64_data: str | None = None) -> str:
    """Chuyển đổi file .md, .txt, .csv, .docx, .xlsx, .pdf thành cấu trúc Markdown chuẩn để cắt chunk."""
    ext = Path(filename).suffix.lower()
    stem = Path(filename).stem.replace("_", " ").title()
    raw = base64.b64decode(base64_data) if base64_data else content.encode("utf-8", errors="ignore")

    if ext == ".docx" and base64_data:
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                xml_bytes = zf.read("word/document.xml")
            root = ET.fromstring(xml_bytes)
            paras = []
            for p in root.iter():
                if p.tag.endswith("}p"):
                    texts = [t.text for t in p.iter() if t.tag.endswith("}t") and t.text]
                    line = "".join(texts).strip()
                    if line:
                        paras.append(line)
            body = "\n\n".join(paras)
            return f"# {stem}\n\n## Nội dung tài liệu Word\n{body}\n"
        except Exception:
            pass

    if ext == ".xlsx" and base64_data:
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                shared = []
                if "xl/sharedStrings.xml" in zf.namelist():
                    s_root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
                    for si in s_root.iter():
                        if si.tag.endswith("}si"):
                            shared.append("".join(t.text or "" for t in si.iter() if t.tag.endswith("}t")))
                sheet_xml = zf.read("xl/worksheets/sheet1.xml")
                sh_root = ET.fromstring(sheet_xml)
                rows_out = []
                for row in sh_root.iter():
                    if row.tag.endswith("}row"):
                        vals = []
                        for c in row:
                            t_attr = c.attrib.get("t")
                            v_node = next(( ch for ch in c if ch.tag.endswith("}v") ), None)
                            if v_node is not None and v_node.text is not None:
                                val = shared[int(v_node.text)] if t_attr == "s" and int(v_node.text) < len(shared) else v_node.text
                                vals.append(str(val).strip())
                        if vals:
                            rows_out.append(" | ".join(vals))
            return f"# {stem}\n\n## Bảng dữ liệu Excel\n" + "\n".join(rows_out) + "\n"
        except Exception:
            pass

    if ext == ".csv":
        text_csv = raw.decode("utf-8", errors="ignore") if base64_data else content
        reader = csv.reader(io.StringIO(text_csv))
        lines = [" | ".join(row) for row in reader if any(cell.strip() for cell in row)]
        return f"# {stem}\n\n## Dữ liệu bảng CSV\n" + "\n".join(lines) + "\n"

    if ext == ".pdf" and base64_data:
        # Trích xuất text streams từ PDF nếu không cài thư viện ngoài
        text_matches = re.findall(rb"\(([^\(\)]{3,200})\)", raw)
        extracted = " ".join(m.decode("latin1", errors="ignore") for m in text_matches).strip()
        if not extracted and content:
            extracted = content
        return f"# {stem}\n\n## Nội dung PDF\n{extracted or 'Tài liệu PDF đã tải lên.'}\n"

    text_out = content.strip() or raw.decode("utf-8", errors="ignore").strip()
    if not text_out.startswith("#"):
        text_out = f"# {stem}\n\n## Nội dung\n{text_out}"
    return text_out + "\n"


@dataclass
class Chunk:
    id: str
    source: str
    title: str
    text: str


class BM25Retriever:
    """Hybrid Retriever kết hợp:
    1) BM25 Lexical Score (với mở rộng từ đồng nghĩa tiếng Việt)
    2) Dense Vector / Char N-gram Cosine Similarity
    3) Cross-Encoder Reranker (ưu tiên khớp cụm từ chính xác & tiêu đề mục)
    """

    def __init__(self, kb_dir: Path | None = None, k1: float = 1.5, b: float = 0.75):
        if kb_dir is None:
            from .shop import active

            kb_dir = active().kb_dir
        self.kb_dir = kb_dir
        self.k1, self.b = k1, b
        self.chunks: list[Chunk] = []
        self._tf: list[Counter] = []
        self._tri: list[Counter] = []
        self._df: Counter = Counter()
        self._avgdl = 0.0
        self._emb: list[list[float]] = []
        self.embedding_mode = "char-3gram"
        self.reload()

    @staticmethod
    def _embed(texts: list[str]) -> list[list[float]]:
        """Gọi API /embeddings tương thích OpenAI (bge-m3, multilingual-e5, text-embedding-3...)."""
        import json as _json
        import urllib.request as _rq
        req = _rq.Request(
            f"{config.EMBEDDING_BASE_URL.rstrip('/')}/embeddings",
            data=_json.dumps({"model": config.EMBEDDING_MODEL, "input": texts}).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {config.EMBEDDING_API_KEY}"},
            method="POST",
        )
        with _rq.urlopen(req, timeout=20) as resp:
            data = _json.loads(resp.read().decode("utf-8"))
        return [d["embedding"] for d in sorted(data["data"], key=lambda d: d.get("index", 0))]

    @staticmethod
    def _dense_cos(a: list[float], b: list[float]) -> float:
        num = sum(x * y for x, y in zip(a, b))
        da = math.sqrt(sum(x * x for x in a)) or 1.0
        db = math.sqrt(sum(y * y for y in b)) or 1.0
        return num / (da * db)

    def reload(self) -> None:
        self.chunks = []
        for path in sorted(self.kb_dir.glob("*.md")):
            self.chunks.extend(self._split(path))
        self._tf = [Counter(tokenize(c.title + " " + c.text)) for c in self.chunks]
        self._tri = [_char_trigrams(c.title + " " + c.text) for c in self.chunks]
        self._df = Counter()
        for tf in self._tf:
            self._df.update(tf.keys())
        self._avgdl = sum(sum(tf.values()) for tf in self._tf) / max(1, len(self._tf))
        self._emb, self.embedding_mode = [], "char-3gram"
        if config.EMBEDDING_MODEL and self.chunks:
            try:
                self._emb = self._embed([c.title + "\n" + c.text for c in self.chunks])
                self.embedding_mode = f"dense:{config.EMBEDDING_MODEL}"
            except Exception:
                self._emb = []

    @staticmethod
    def _windows(body: str) -> list[str]:
        """Cắt đoạn dài thành các cửa sổ theo câu (~CHUNK_MAX_CHARS), gối lên nhau 1 câu để không mất ngữ cảnh."""
        if len(body) <= CHUNK_MAX_CHARS:
            return [body]
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", body) if s.strip()]
        out, cur = [], []
        for s in sentences:
            if cur and len(" ".join(cur + [s])) > CHUNK_MAX_CHARS:
                out.append(" ".join(cur))
                cur = cur[-CHUNK_OVERLAP_SENTENCES:] if CHUNK_OVERLAP_SENTENCES else []
            cur.append(s)
        if cur:
            out.append(" ".join(cur))
        return out

    @staticmethod
    def _split(path: Path) -> list[Chunk]:
        chunks, title, buf = [], path.stem, []
        doc_title = path.stem

        def flush():
            body = "\n".join(buf).strip()
            if not body:
                return
            parts = BM25Retriever._windows(body)
            for k, part in enumerate(parts):
                t = title if len(parts) == 1 else f"{title} ({k + 1}/{len(parts)})"
                chunks.append(Chunk(f"{path.stem}#{len(chunks)}", path.name, t, part))

        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                doc_title = line[2:].strip()
                title = doc_title
            elif line.startswith("## "):
                flush()
                buf = []
                title = f"{doc_title} - {line[3:].strip()}"
            else:
                buf.append(line)
        flush()
        return chunks

    def _rerank_bonus(self, query: str, chunk: Chunk) -> float:
        q_norm = normalize(query).strip(" ?!.")
        c_title = normalize(chunk.title)
        c_text = normalize(chunk.text)
        bonus = 0.0
        q_words = [w for w in re.findall(r"[a-z0-9]+", q_norm) if w not in _STOPWORDS]
        if not q_words:
            return 0.0
        # So khớp nguyên từ (tránh "o" khớp nhầm mọi chữ có chứa "o")
        title_words = set(re.findall(r"[a-z0-9]+", c_title))
        title_hits = sum(1 for w in q_words if w in title_words)
        bonus += 0.4 * (title_hits / len(q_words))
        # thưởng nếu có cụm 2 từ liên tiếp khớp nguyên văn trong văn bản
        padded_text = f" {' '.join(re.findall(r'[a-z0-9]+', c_text))} "
        padded_title = f" {' '.join(re.findall(r'[a-z0-9]+', c_title))} "
        for a, b in zip(q_words, q_words[1:]):
            if f" {a} {b} " in padded_text or f" {a} {b} " in padded_title:
                bonus += 0.25
        return min(bonus, 1.2)

    def search(self, query: str, top_k: int | None = None, min_score: float | None = None) -> list[dict]:
        top_k = top_k or config.RAG_TOP_K
        min_score = config.RAG_MIN_SCORE if min_score is None else min_score
        syn = _synonyms()
        q = tokenize(query, expand_synonyms=True)
        q_words = list(dict.fromkeys(t for t in tokenize(query) if "_" not in t))
        q_tri = _char_trigrams(query)
        q_emb = None
        if self._emb:
            try:
                q_emb = self._embed([query])[0]
            except Exception:
                q_emb = None
        n = len(self.chunks)
        scored = []
        for i, tf in enumerate(self._tf):
            dl = sum(tf.values())
            bm25 = 0.0
            for term in q:
                if term not in tf:
                    continue
                idf = math.log(1 + (n - self._df[term] + 0.5) / (self._df[term] + 0.5))
                f = tf[term]
                bm25 += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self._avgdl))
            if q_emb is not None:
                # Dense embedding hiểu đồng nghĩa nên được tính cả khi BM25 thấp (vd "áo bị chật" ~ "đổi size")
                dense = self._dense_cos(q_emb, self._emb[i])
                vec_sim = dense * 2.0 if dense >= 0.5 else 0.0
            else:
                vec_sim = _cosine(q_tri, self._tri[i]) if bm25 > 0.6 else 0.0
            rerank = self._rerank_bonus(query, self.chunks[i]) if bm25 > 0.6 else 0.0
            total = bm25 + 1.2 * vec_sim + rerank
            # Chặn khớp nhầm do bỏ dấu ("máy" ~ "may"): phải phủ đủ tỉ lệ từ của câu hỏi (trừ khi dùng dense embedding)
            covered = sum(1 for w in q_words if w in tf or any(s in tf for s in syn.get(w, ())))
            if q_emb is None and q_words and covered / len(q_words) < MIN_QUERY_COVERAGE:
                continue
            if total >= min_score:
                scored.append((total, bm25, vec_sim, rerank, i))
        scored.sort(reverse=True)
        return [
            {
                "id": self.chunks[i].id,
                "source": self.chunks[i].source,
                "title": self.chunks[i].title,
                "text": self.chunks[i].text,
                "score": round(total, 3),
                "bm25_score": round(bm25, 3),
                "vector_score": round(vec_sim, 3),
                "rerank_score": round(rerank, 3),
            }
            for total, bm25, vec_sim, rerank, i in scored[:top_k]
        ]
