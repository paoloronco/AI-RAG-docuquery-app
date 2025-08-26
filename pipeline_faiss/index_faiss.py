import os, sys, re, json, gc, hashlib, time
from pathlib import Path
from typing import Iterator, Tuple, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# ============ CONFIG ============
EMBED_MODEL = "intfloat/e5-small-v2"
OUT_DIR = "./faiss_index"
CHUNK_SIZE = 800          # larger chunks = fewer pieces
CHUNK_OVERLAP = 120
BATCH_GPU = 64
BATCH_CPU = 16
MAX_PAGES_PER_PDF = 2000
MAX_CHUNKS_PER_PDF = 20000
PERSIST_EVERY = 5000      # save the index every N vectors
# ================================

def log(msg: str):
    print(msg, flush=True)

def sanitize(s: str) -> str:
    s = s.replace("\x00", " ")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()

def iter_pages(pdf_path: Path) -> Iterator[Tuple[int, str]]:
    """Extract text page by page (pypdf, no cumulative memory)."""
    try:
        reader = PdfReader(str(pdf_path))
    except Exception as e:
        log(f"[SKIP] {pdf_path.name}: failed to open ({e})")
        return
    for i, page in enumerate(reader.pages):
        if i >= MAX_PAGES_PER_PDF:
            log("  (page limit reached)")
            break
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        t = sanitize(t)
        if t:
            yield i, t

def chunker(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    n, i = len(text), 0
    while i < n:
        j = min(i + size, n)
        yield text[i:j]
        i = max(0, j - overlap)

def open_meta_writer():
    os.makedirs(OUT_DIR, exist_ok=True)
    return open(os.path.join(OUT_DIR, "meta.jsonl"), "w", encoding="utf-8")

def main(folder: str):
    os.makedirs(OUT_DIR, exist_ok=True)

    # embedder on CUDA if available (only for speed); RAM remains low because we stream
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_grad_enabled(False)
        torch.set_num_threads(1)
        if device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
    except Exception:
        device = "cpu"

    embed = SentenceTransformer(EMBED_MODEL, device=device)
    log(f"Embedder: {EMBED_MODEL} on {device}")

    # prepare (or reset) FAISS index + meta.jsonl
    idx_path = os.path.join(OUT_DIR, "index.faiss")
    if os.path.exists(idx_path):
        os.remove(idx_path)
    meta_f = open_meta_writer()

    index = None
    dim = None
    total_added = 0
    batch_size = BATCH_GPU if device == "cuda" else BATCH_CPU

    def add_batch(texts: List[str], metas: List[dict]):
        """Embed + add on the fly; immediately write meta; no RAM accumulation."""
        nonlocal index, dim, total_added, batch_size
        if not texts:
            return
        # e5 requires prefixes
        inps = [f"passage: {t}" for t in texts]
        # GPU out-of-memory backoff
        while True:
            try:
                vecs = embed.encode(inps, normalize_embeddings=True, show_progress_bar=False)
                vecs = vecs.astype("float32")
                break
            except RuntimeError as e:
                msg = str(e).lower()
                if "cuda" in msg and "out of memory" in msg and batch_size > 1:
                    batch_size = max(1, batch_size // 2)
                    log(f"  [OOM] reducing batch to {batch_size} and retrying…")
                    half = max(1, len(inps) // 2)
                    add_batch(inps[:half], metas[:half])
                    add_batch(inps[half:], metas[half:])
                    return
                else:
                    raise

        # create index on the first batch (with correct dimension)
        nonlocal dim
        if index is None:
            dim = vecs.shape[1]
            index = faiss.IndexFlatIP(dim)

        index.add(vecs)
        # immediately write meta (one line per vector, same order)
        for m in metas:
            meta_f.write(json.dumps(m, ensure_ascii=False) + "\n")
        meta_f.flush()

        total_added += vecs.shape[0]
        if total_added % PERSIST_EVERY == 0:
            faiss.write_index(index, idx_path)
            log(f"  [checkpoint] total vectors: {total_added}")

        del vecs
        gc.collect()

    pdfs = list(Path(folder).glob("**/*.pdf"))
    if not pdfs:
        log("No PDFs found.")
        meta_f.close()
        return

    for p in pdfs:
        log(f"Indexing: {p.name}")
        stat = p.stat()
        base_id = hashlib.sha1(f"{p}|{int(stat.st_mtime)}".encode()).hexdigest()

        cur_texts: List[str] = []
        cur_metas: List[dict] = []
        chunks_emitted = 0

        for pg, text in iter_pages(p):
            for ch in chunker(text):
                if chunks_emitted >= MAX_CHUNKS_PER_PDF:
                    log("  (chunk limit reached)")
                    break
                chunk_text = ch.strip()
                if not chunk_text:
                    continue
                cur_texts.append(chunk_text)
                cur_metas.append({"file": str(p), "page": pg, "text": chunk_text})
                chunks_emitted += 1

                if len(cur_texts) >= batch_size:
                    add_batch(cur_texts, cur_metas)
                    cur_texts, cur_metas = [], []
            if chunks_emitted >= MAX_CHUNKS_PER_PDF:
                break

        if cur_texts:
            add_batch(cur_texts, cur_metas)
            cur_texts, cur_metas = [], []

        log(f"  → chunks indexed: {chunks_emitted}")

    if index is not None:
        faiss.write_index(index, idx_path)
    meta_f.close()

    log(f"✅ DONE. Total vectors: {total_added}. Index: {idx_path}")
    log("Note: if search is slow, you can try faiss.IndexHNSWFlat or IVF in the future.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python index_faiss.py <pdf_folder>")
        raise SystemExit(1)
    main(sys.argv[1])
