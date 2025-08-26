import os, sys, json, re
from pathlib import Path

os.environ["OMP_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
os.environ["TOKENIZERS_PARALLELISM"]="false"

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QLabel, QFileDialog, QMessageBox
)

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "intfloat/e5-small-v2"
FAISS_DIR   = "./faiss_index"
IDX_PATH    = os.path.join(FAISS_DIR, "index.faiss")
META_PATH   = os.path.join(FAISS_DIR, "meta.jsonl")

def load_meta():
    metas=[]
    with open(META_PATH, "r", encoding="utf-8") as f:
        for line in f:
            metas.append(json.loads(line))
    return metas

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RAG (FAISS) – chat with your PDFs")
        self.resize(900,650)

        self.idx=None
        self.metas=[]
        # query embedder on CPU: fast and no tricky drivers
        self.embed_q = SentenceTransformer(EMBED_MODEL, device="cpu")

        lay=QVBoxLayout(self)

        top=QHBoxLayout()
        self.lblIdx=QLabel(f"Index: {Path(IDX_PATH).resolve() if os.path.exists(IDX_PATH) else '— not found —'}")
        top.addWidget(self.lblIdx)

        self.bReload=QPushButton("Reload index")
        self.bReload.clicked.connect(self.reload_index)
        top.addWidget(self.bReload)

        self.bPick=QPushButton("Open index folder…")
        self.bPick.clicked.connect(self.pick_dir)
        top.addWidget(self.bPick)

        lay.addLayout(top)

        self.inp=QTextEdit(); self.inp.setPlaceholderText("Write a question…")
        lay.addWidget(self.inp)

        self.bAsk=QPushButton("Search and answer")
        self.bAsk.clicked.connect(self.ask)
        lay.addWidget(self.bAsk)

        self.out=QTextEdit(); self.out.setReadOnly(True)
        lay.addWidget(self.out)

        # autoload if available
        self.reload_index(initial=True)

    def pick_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select FAISS index folder (must contain index.faiss and meta.jsonl)")
        if not d: return
        global FAISS_DIR, IDX_PATH, META_PATH
        FAISS_DIR = d
        IDX_PATH  = os.path.join(FAISS_DIR, "index.faiss")
        META_PATH = os.path.join(FAISS_DIR, "meta.jsonl")
        self.lblIdx.setText(f"Index: {IDX_PATH}")
        self.reload_index()

    def reload_index(self, initial=False):
        try:
            self.idx  = faiss.read_index(IDX_PATH)
            self.metas= load_meta()
            self.lblIdx.setText(f"Index: {IDX_PATH}  •  vectors={self.idx.ntotal}")
            if not initial:
                self.out.setPlainText("✅ Index reloaded.")
        except Exception as e:
            self.idx=None; self.metas=[]
            msg = f"⚠ Cannot find a valid index in {FAISS_DIR}\n{e}"
            if initial:
                self.out.append(msg)
            else:
                QMessageBox.warning(self,"Missing index", msg)

    def ask(self):
        q = self.inp.toPlainText().strip()
        if not q:
            QMessageBox.warning(self,"Warning","Write a question.")
            return
        if self.idx is None or not self.metas:
            QMessageBox.warning(self,"Warning","Reload or create the index (index_faiss.py).")
            return

        # embed query (e5 → prefix "query: ")
        qv = self.embed_q.encode([f"query: {q}"], normalize_embeddings=True, show_progress_bar=False).astype("float32")
        D,I = self.idx.search(qv, k=5)

        parts=[]
        for rank,(idx_id,score) in enumerate(zip(I[0], D[0]), start=1):
            m = self.metas[idx_id]
            src = os.path.basename(m.get("file","?"))
            pg  = m.get("page","?")
            txt = (m.get("text") or "").strip().replace("\n"," ")
            if not txt:
                txt = "⚠ Snippet not available. Re-index with updated index_faiss.py (meta['text'])."
            parts.append(f"[{rank}] {src} / page {pg} • score={score:.3f}\n{txt[:700]}{'...' if len(txt)>700 else ''}\n")

        if not parts:
            self.out.setPlainText("No results.")
            return

        answer = (
            "Answer built ONLY from the retrieved passages.\n\n" +
            "\n".join(parts) +
            f"\n—\nQUESTION: {q}\nSUMMARY (use the passages above): "
        )
        self.out.setPlainText(answer)

def main():
    app=QApplication(sys.argv)
    w=App(); w.show()
    sys.exit(app.exec())

if __name__=="__main__":
    main()
