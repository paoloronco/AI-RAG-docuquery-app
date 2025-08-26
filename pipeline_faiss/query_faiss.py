import os, json, faiss, numpy as np
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "intfloat/e5-small-v2"
OUT_DIR = "./faiss_index"

def load_meta():
    metas = []
    with open(os.path.join(OUT_DIR, "meta.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            metas.append(json.loads(line))
    return metas

def main():
    idx = faiss.read_index(os.path.join(OUT_DIR, "index.faiss"))
    metas = load_meta()
    emb = SentenceTransformer(EMBED_MODEL, device="cpu")

    while True:
        q = input("\nQuestion (press Enter to exit): ").strip()
        if not q: break
        qv = emb.encode([f"query: {q}"], normalize_embeddings=True, show_progress_bar=False).astype("float32")
        D, I = idx.search(qv, k=5)
        print("\n— Possible ANSWERS —")
        for rank, (idx_id, score) in enumerate(zip(I[0], D[0]), start=1):
            m = metas[idx_id]
            print(f"[{rank}] score={score:.3f} file={os.path.basename(m['file'])} page={m['page']}")
            print(m['text'][:400].replace("\n"," ") + ("..." if len(m['text'])>400 else ""))
            print()

if __name__ == "__main__":
    main()
