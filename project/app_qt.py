from __future__ import annotations
import os, sys, json
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QFileDialog, QProgressBar, QComboBox, QMessageBox, QLineEdit, QTextBrowser,
    QDialog, QFormLayout, QDialogButtonBox, QCheckBox
)
from PyQt6.QtCore import QThread, pyqtSignal, QUrl
from PyQt6.QtGui import QDesktopServices

from indexer import Indexer
from retrieve import Retriever
from llm_clients import NoLLM, OpenAIChat, HFLocal
from config import EMBED_MODELS, INDEX_TYPES, LLM_BACKENDS, DEFAULTS, IndexConfig

INDEX_DIR = Path("./faiss_index").resolve()

# ---------------- app config helpers ----------------
def app_config_path() -> Path:
    if os.name == "nt":
        base = Path(os.getenv("APPDATA", str(Path.home() / "AppData" / "Roaming")))
        return base / "RAG-Pro" / "config.json"
    return Path.home() / ".rag-pro" / "config.json"

def load_config() -> dict:
    p = app_config_path()
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}

def save_config(cfg: dict) -> None:
    p = app_config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------------- Index worker ----------------
class IndexThread(QThread):
    status = pyqtSignal(str); progress = pyqtSignal(int); done = pyqtSignal(int, int)
    def __init__(self, folder: Path, cfg: IndexConfig):
        super().__init__(); self.folder = folder; self.cfg = cfg; self._cancel = False
    def cancel(self): self._cancel = True
    def run(self):
        def should_cancel(): return self._cancel
        idx = Indexer(INDEX_DIR, self.cfg, on_status=self.status.emit, on_progress=self.progress.emit, should_cancel=should_cancel)
        files, vecs = idx.build(self.folder); self.done.emit(files, vecs)

# ---------------- Ask worker ----------------
class AskThread(QThread):
    error = pyqtSignal(str); ready = pyqtSignal(str)
    def __init__(self, question: str, retriever: Retriever, llm):
        super().__init__(); self.q = question; self.retriever = retriever; self.llm = llm
    def run(self):
        try:
            hits = self.retriever.search(self.q); best = self.retriever.best_score(hits); ctx = self.retriever.gather(hits)
            if best < DEFAULTS["min_sim"]:
                html = "<b>No confident answer found in the documents.</b><br><br>Closest passages:<br><pre>"+self.retriever.format_context(ctx)+"</pre>"
                self.ready.emit(html); return
            context = self.retriever.format_context(ctx)
            system=("You are a RAG assistant. Answer ONLY using the provided passages. Cite sources as [number]. "
                    "If the information is missing, say 'Not found in the documents'.")
            user=(f"QUESTION: {self.q}\n\nPASSAGES:\n{context}\n\nInstructions: write a concise answer in English, including references [1], [2]…")
            answer = self.llm.generate(system, user, max_tokens=600)
            def linkify(p:dict)->str:
                path=Path(p.get("file","")).resolve()
                url=QUrl.fromLocalFile(str(path))
                return f"<a href='{url.toString()}' title='{path}'>Open</a>"
            cites=[f"[{p['rank']}] {os.path.basename(p.get('file','?'))} • page {p.get('page','?')} • score={p['score']:.3f} • {linkify(p)}" for p in ctx]
            html = f"<b>Answer</b><br><div style='white-space:pre-wrap'>{answer}</div><br><b>Sources</b><br>"+"<br>".join(cites)
            self.ready.emit(html)
        except Exception as e:
            self.error.emit(str(e))

# ---------------- OpenAI auth dialog ----------------
OPENAI_PRESETS = [
    "gpt-4o-mini",
    "gpt-4o",
    "o4-mini",
    "gpt-4.1-mini",
    "gpt-3.5-turbo",
    "Other…",
]

class OpenAIDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configure OpenAI / compatible")
        self.eKey = QLineEdit(); self.eKey.setEchoMode(QLineEdit.EchoMode.Password); self.eKey.setPlaceholderText("sk-...")
        self.eUrl = QLineEdit(); self.eUrl.setPlaceholderText("https://your-endpoint/v1  (optional)")
        self.cbRemember = QCheckBox("Remember for next time (save to local config)")
        self.cbModel = QComboBox(); [self.cbModel.addItem(m) for m in OPENAI_PRESETS]
        self.eCustomModel = QLineEdit(); self.eCustomModel.setPlaceholderText("Custom model name…"); self.eCustomModel.setEnabled(False)

        # prefill from env or saved config
        cfg = load_config()
        self.eKey.setText(os.getenv("OPENAI_API_KEY", cfg.get("OPENAI_API_KEY", "")))
        self.eUrl.setText(os.getenv("OPENAI_BASE_URL", cfg.get("OPENAI_BASE_URL", "")))
        preset = os.getenv("OPENAI_MODEL", cfg.get("OPENAI_MODEL", "gpt-4o-mini"))
        if preset in OPENAI_PRESETS:
            self.cbModel.setCurrentText(preset)
        else:
            self.cbModel.setCurrentText("Other…")
            self.eCustomModel.setEnabled(True)
            self.eCustomModel.setText(preset)

        def on_model_change(idx:int):
            self.eCustomModel.setEnabled(self.cbModel.currentText() == "Other…")
        self.cbModel.currentIndexChanged.connect(on_model_change)

        form = QFormLayout(self)
        form.addRow("API key", self.eKey)
        form.addRow("Base URL", self.eUrl)
        form.addRow("Model", self.cbModel)
        form.addRow("", self.eCustomModel)
        form.addRow(self.cbRemember)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        form.addRow(btns)

    def values(self):
        model = self.cbModel.currentText()
        if model == "Other…":
            model = self.eCustomModel.text().strip()
        return self.eKey.text().strip(), self.eUrl.text().strip(), model.strip(), self.cbRemember.isChecked()

# ---------------- App ----------------
class App(QWidget):
    def __init__(self):
        super().__init__(); self.setWindowTitle("RAG Pro — FAISS GUI"); self.resize(1000, 720)
        self.retriever: Retriever | None = None; self.llm_backend = "none"; self.llm = NoLLM()

        tabs = QTabWidget(self); v = QVBoxLayout(self); v.addWidget(tabs)

        # Tab: Indexing
        w_idx = QWidget(); tabs.addTab(w_idx, "Indexing"); l = QVBoxLayout(w_idx)
        row = QHBoxLayout(); l.addLayout(row)
        self.eFolder = QLineEdit(); self.eFolder.setPlaceholderText("Dataset folder…")
        bPick = QPushButton("Browse…"); bPick.clicked.connect(self.pick_folder)
        row.addWidget(QLabel("Folder:")); row.addWidget(self.eFolder,1); row.addWidget(bPick)

        row2 = QHBoxLayout(); l.addLayout(row2)
        self.cbEmbed = QComboBox(); [self.cbEmbed.addItem(v["display"], k) for k,v in EMBED_MODELS.items()]
        self.cbIndex = QComboBox(); [self.cbIndex.addItem(lbl, key) for key,lbl in INDEX_TYPES]
        row2.addWidget(QLabel("Embedding:")); row2.addWidget(self.cbEmbed)
        row2.addWidget(QLabel("Index:")); row2.addWidget(self.cbIndex)

        row3 = QHBoxLayout(); l.addLayout(row3)
        self.bIndex = QPushButton("Build index"); self.bIndex.clicked.connect(self.start_index)
        self.bCancel= QPushButton("Cancel"); self.bCancel.clicked.connect(self.cancel_index); self.bCancel.setEnabled(False)
        row3.addWidget(self.bIndex); row3.addWidget(self.bCancel)

        self.prog = QProgressBar(); self.prog.setValue(0); l.addWidget(self.prog)
        self.log = QTextEdit(); self.log.setReadOnly(True); l.addWidget(self.log,1)

        # Tab: Chat
        w_chat = QWidget(); tabs.addTab(w_chat, "Chat"); c = QVBoxLayout(w_chat)
        top = QHBoxLayout(); c.addLayout(top)
        self.lblIndex = QLabel(self.index_status_text())
        self.bReload = QPushButton("Reload index"); self.bReload.clicked.connect(self.reload_index)
        top.addWidget(self.lblIndex); top.addWidget(self.bReload)

        rowm = QHBoxLayout(); c.addLayout(rowm)
        self.cbLLM = QComboBox(); [self.cbLLM.addItem(lbl, key) for key,lbl in LLM_BACKENDS]
        self.cbLLM.currentIndexChanged.connect(self.on_backend_change)
        self.eModelName = QLineEdit(); self.eModelName.setPlaceholderText("Model (e.g. gpt-4o-mini / Qwen…)")
        rowm.addWidget(QLabel("LLM backend:")); rowm.addWidget(self.cbLLM)
        rowm.addWidget(QLabel("Model:")); rowm.addWidget(self.eModelName,1)
        bSet = QPushButton("Apply"); bSet.clicked.connect(self.apply_llm); rowm.addWidget(bSet)

        self.inp = QTextEdit(); self.inp.setPlaceholderText("Type your question…"); c.addWidget(self.inp)
        self.bAsk = QPushButton("Search and Answer"); self.bAsk.clicked.connect(self.ask); c.addWidget(self.bAsk)
        self.out = QTextBrowser()
        # enable opening local files by clicking <a href="file:///...">
        self.out.setOpenExternalLinks(False)
        self.out.setOpenLinks(False)
        self.out.anchorClicked.connect(QDesktopServices.openUrl)
        c.addWidget(self.out,1)

        self.worker: IndexThread | None = None; self.asker: AskThread | None = None

    # ---------- when backend changes ----------
    def on_backend_change(self, _idx:int):
        kind = self.cbLLM.currentData()
        # Disable/enable "Model" field depending on backend
        self.eModelName.setEnabled(kind != "openai")

        if kind == "openai":
            dlg = OpenAIDialog(self)
            if dlg.exec() == QDialog.DialogCode.Accepted:
                key, url, model, remember = dlg.values()
                if not key:
                    QMessageBox.warning(self, "Missing API key", "Please enter a valid API key.")
                    self.cbLLM.setCurrentIndex(0); return
                if not model:
                    QMessageBox.warning(self, "Missing model", "Please select or type a model name.")
                    self.cbLLM.setCurrentIndex(0); return

                # Set for current session
                os.environ["OPENAI_API_KEY"] = key
                os.environ["OPENAI_MODEL"] = model
                if url: os.environ["OPENAI_BASE_URL"] = url
                else: os.environ.pop("OPENAI_BASE_URL", None)

                # Optional persist
                if remember:
                    cfg = load_config()
                    cfg["OPENAI_API_KEY"] = key
                    if url: cfg["OPENAI_BASE_URL"] = url
                    else: cfg.pop("OPENAI_BASE_URL", None)
                    cfg["OPENAI_MODEL"] = model
                    save_config(cfg)

                # Apply backend immediately
                try:
                    self.llm = OpenAIChat(model=model)
                    self.llm_backend = "openai"
                    QMessageBox.information(self, "LLM", f"Applied: {self.llm.name}")
                except Exception as e:
                    QMessageBox.warning(self, "LLM error", str(e))
                    self.cbLLM.setCurrentIndex(0)
            else:
                # cancelled → back to "No LLM"
                self.cbLLM.setCurrentIndex(0)
                self.apply_llm()
        else:
            # For other backends apply immediately (including "No LLM")
            self.apply_llm()

    # ---------- Indexing ----------
    def pick_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select a folder to index")
        if d: self.eFolder.setText(d)

    def start_index(self):
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "Please wait", "Indexing in progress…"); return
        folder = self.eFolder.text().strip()
        if not folder:
            QMessageBox.warning(self, "Attention", "Please choose a folder."); return
        cfg = IndexConfig(embed_model=self.cbEmbed.currentData(), index_type=self.cbIndex.currentData())
        self.worker = IndexThread(Path(folder), cfg)
        self.worker.status.connect(self.log.append); self.worker.progress.connect(self.on_progress_update); self.worker.done.connect(self.index_done)
        self.bIndex.setEnabled(False); self.bCancel.setEnabled(True)
        self.log.append(f"\n— Starting indexing → {folder}"); self.prog.setRange(0,0); self.worker.start()

    def on_progress_update(self, p:int):
        if self.prog.minimum()==0 and self.prog.maximum()==0: self.prog.setRange(0,100)
        self.prog.setValue(p)

    def cancel_index(self):
        if self.worker: self.worker.cancel()

    def index_done(self, files:int, vecs:int):
        self.bIndex.setEnabled(True); self.bCancel.setEnabled(False); self.prog.setRange(0,100); self.prog.setValue(100)
        self.log.append(f"\n✅ Completed. Files indexed: {files} • Vectors: {vecs}")
        self.lblIndex.setText(self.index_status_text()); self.retriever=None

    # ---------- Chat ----------
    def index_status_text(self) -> str:
        idx = INDEX_DIR/"index.faiss"; return f"Index: {idx}" if idx.exists() else "Index: — not found —"

    def reload_index(self):
        try:
            self.retriever = Retriever(INDEX_DIR)
            self.lblIndex.setText(self.index_status_text()+f"  • vectors={self.retriever.idx.ntotal}  • embed={self.retriever.embed_model}")
            QMessageBox.information(self, "OK", "Index reloaded.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not find a valid index.\n{e}")

    def apply_llm(self):
        kind = self.cbLLM.currentData(); name = self.eModelName.text().strip() or ""
        try:
            if kind=="none":
                self.llm = NoLLM(); self.llm_backend = "none"
            elif kind=="openai":
                model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                self.llm = OpenAIChat(model=model); self.llm_backend = "openai"
            elif kind=="hf_local":
                self.llm = HFLocal(model_id=name or "Qwen/Qwen2.5-0.5B-Instruct"); self.llm_backend = "hf_local"
            QMessageBox.information(self, "LLM", f"Applied: {self.llm.name}")
        except Exception as e:
            QMessageBox.warning(self, "LLM error", str(e))
            self.llm = NoLLM(); self.llm_backend = "none"

    def ask(self):
        q = self.inp.toPlainText().strip()
        if not q:
            QMessageBox.warning(self, "Empty field", "Please type a question."); return
        if self.retriever is None:
            try: self.retriever = Retriever(INDEX_DIR)
            except Exception:
                QMessageBox.warning(self, "Missing index", "Reload or build the index first."); return
        if self.asker and self.asker.isRunning(): return
        self.bAsk.setEnabled(False); self.out.setHtml("<i>Searching documents…</i>")
        self.asker = AskThread(q, self.retriever, self.llm); self.asker.ready.connect(self.on_answer_ready); self.asker.error.connect(self.on_answer_error); self.asker.start()

    def on_answer_ready(self, html: str):
        self.out.setHtml(html); self.bAsk.setEnabled(True); self.inp.clear(); self.inp.setFocus()

    def on_answer_error(self, msg: str):
        QMessageBox.warning(self, "Error while answering", msg); self.bAsk.setEnabled(True); self.inp.setFocus()

if __name__ == "__main__":
    app = QApplication(sys.argv); w = App(); w.show(); sys.exit(app.exec())
