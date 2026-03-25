from __future__ import annotations
import os, sys, json
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QFileDialog, QProgressBar, QComboBox, QMessageBox, QLineEdit, QTextBrowser,
    QDialog, QFormLayout, QDialogButtonBox, QCheckBox
)
from PyQt6.QtCore import QThread, pyqtSignal, QUrl, QByteArray
from PyQt6.QtGui import QDesktopServices, QIcon, QPixmap

# ── embedded app icon (256×256 PNG, base64) ──────────────────────────────────
_ICON_B64 = (
    "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"
)

def _app_icon() -> QIcon:
    data = QByteArray.fromBase64(_ICON_B64.encode())
    px = QPixmap()
    px.loadFromData(data, "PNG")
    return QIcon(px)

from indexer import Indexer
from retrieve import Retriever
from llm_clients import NoLLM, OpenAIChat, AnthropicChat, HFLocal
from config import EMBED_MODELS, INDEX_TYPES, LLM_BACKENDS, DEFAULTS, IndexConfig

# ---------------- app config / index helpers ----------------
def _rag_pro_dir() -> Path:
    if os.name == "nt":
        return Path(os.getenv("APPDATA", str(Path.home() / "AppData" / "Roaming"))) / "RAG-Pro"
    return Path.home() / ".rag-pro"

def app_config_path() -> Path:
    return _rag_pro_dir() / "config.json"

def indexes_base_dir() -> Path:
    return _rag_pro_dir() / "indexes"

def list_indexes() -> list[str]:
    base = indexes_base_dir()
    if not base.exists():
        return []
    return sorted([d.name for d in base.iterdir() if d.is_dir() and (d / "index.faiss").exists()])

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
    def __init__(self, folder: Path, index_dir: Path, cfg: IndexConfig):
        super().__init__(); self.folder = folder; self.index_dir = index_dir; self.cfg = cfg; self._cancel = False
    def cancel(self): self._cancel = True
    def run(self):
        def should_cancel(): return self._cancel
        idx = Indexer(self.index_dir, self.cfg, on_status=self.status.emit, on_progress=self.progress.emit, should_cancel=should_cancel)
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

# ---------------- Anthropic auth dialog ----------------
ANTHROPIC_PRESETS = [
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-6",
    "claude-opus-4-6",
    "Other…",
]

class AnthropicDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configure Anthropic Claude")
        self.eKey = QLineEdit(); self.eKey.setEchoMode(QLineEdit.EchoMode.Password); self.eKey.setPlaceholderText("sk-ant-...")
        self.cbRemember = QCheckBox("Remember for next time (save to local config)")
        self.cbModel = QComboBox(); [self.cbModel.addItem(m) for m in ANTHROPIC_PRESETS]
        self.eCustomModel = QLineEdit(); self.eCustomModel.setPlaceholderText("Custom model id…"); self.eCustomModel.setEnabled(False)

        cfg = load_config()
        self.eKey.setText(os.getenv("ANTHROPIC_API_KEY", cfg.get("ANTHROPIC_API_KEY", "")))
        preset = os.getenv("ANTHROPIC_MODEL", cfg.get("ANTHROPIC_MODEL", ANTHROPIC_PRESETS[0]))
        if preset in ANTHROPIC_PRESETS:
            self.cbModel.setCurrentText(preset)
        else:
            self.cbModel.setCurrentText("Other…")
            self.eCustomModel.setEnabled(True)
            self.eCustomModel.setText(preset)

        def on_model_change(idx: int):
            self.eCustomModel.setEnabled(self.cbModel.currentText() == "Other…")
        self.cbModel.currentIndexChanged.connect(on_model_change)

        form = QFormLayout(self)
        form.addRow("API key", self.eKey)
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
        return self.eKey.text().strip(), model.strip(), self.cbRemember.isChecked()


# ---------------- App ----------------
class App(QWidget):
    def __init__(self):
        super().__init__(); self.setWindowTitle("RAG Pro — FAISS GUI"); self.resize(1000, 720)
        self.retriever: Retriever | None = None; self.llm_backend = "none"; self.llm = NoLLM()
        self._history: list[tuple[str, str]] = []; self._current_q = ""

        tabs = QTabWidget(self); v = QVBoxLayout(self); v.addWidget(tabs)

        # Tab: Indexing
        w_idx = QWidget(); tabs.addTab(w_idx, "Indexing"); l = QVBoxLayout(w_idx)
        row = QHBoxLayout(); l.addLayout(row)
        self.eFolder = QLineEdit(); self.eFolder.setPlaceholderText("Dataset folder…")
        bPick = QPushButton("Browse…"); bPick.clicked.connect(self.pick_folder)
        row.addWidget(QLabel("Folder:")); row.addWidget(self.eFolder,1); row.addWidget(bPick)

        row_name = QHBoxLayout(); l.addLayout(row_name)
        self.eIndexName = QLineEdit(); self.eIndexName.setPlaceholderText("Index name (e.g. contracts, manual-2024…)")
        row_name.addWidget(QLabel("Index name:")); row_name.addWidget(self.eIndexName, 1)

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
        self.cbIndexSel = QComboBox()
        bRefreshIdx = QPushButton("↻"); bRefreshIdx.setFixedWidth(30); bRefreshIdx.setToolTip("Refresh index list")
        bRefreshIdx.clicked.connect(self._refresh_index_combo)
        self.bReload = QPushButton("Load index"); self.bReload.clicked.connect(self.reload_index)
        top.addWidget(QLabel("Index:")); top.addWidget(self.cbIndexSel, 1); top.addWidget(bRefreshIdx); top.addWidget(self.bReload)
        self.lblIndex = QLabel("—"); c.addWidget(self.lblIndex)
        self._refresh_index_combo()
        self.cbIndexSel.currentIndexChanged.connect(lambda _: setattr(self, "retriever", None))

        rowm = QHBoxLayout(); c.addLayout(rowm)
        self.cbLLM = QComboBox(); [self.cbLLM.addItem(lbl, key) for key,lbl in LLM_BACKENDS]
        self.cbLLM.currentIndexChanged.connect(self.on_backend_change)
        self.eModelName = QLineEdit(); self.eModelName.setPlaceholderText("Model (e.g. gpt-4o-mini / Qwen…)")
        rowm.addWidget(QLabel("LLM backend:")); rowm.addWidget(self.cbLLM)
        rowm.addWidget(QLabel("Model:")); rowm.addWidget(self.eModelName,1)
        bSet = QPushButton("Apply"); bSet.clicked.connect(self.apply_llm); rowm.addWidget(bSet)

        self.inp = QTextEdit(); self.inp.setPlaceholderText("Type your question…"); c.addWidget(self.inp)
        row_ask = QHBoxLayout(); c.addLayout(row_ask)
        self.bAsk = QPushButton("Search and Answer"); self.bAsk.clicked.connect(self.ask)
        bClear = QPushButton("Clear history"); bClear.clicked.connect(self.clear_history)
        row_ask.addWidget(self.bAsk, 1); row_ask.addWidget(bClear)
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
        # Only HF local uses the free-text model field
        self.eModelName.setEnabled(kind == "hf_local")

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
                os.environ["OPENAI_API_KEY"] = key
                os.environ["OPENAI_MODEL"] = model
                if url: os.environ["OPENAI_BASE_URL"] = url
                else: os.environ.pop("OPENAI_BASE_URL", None)
                if remember:
                    cfg = load_config()
                    cfg["OPENAI_API_KEY"] = key
                    if url: cfg["OPENAI_BASE_URL"] = url
                    else: cfg.pop("OPENAI_BASE_URL", None)
                    cfg["OPENAI_MODEL"] = model
                    save_config(cfg)
                try:
                    self.llm = OpenAIChat(model=model)
                    self.llm_backend = "openai"
                    QMessageBox.information(self, "LLM", f"Applied: {self.llm.name}")
                except Exception as e:
                    QMessageBox.warning(self, "LLM error", str(e))
                    self.cbLLM.setCurrentIndex(0)
            else:
                self.cbLLM.setCurrentIndex(0)
                self.apply_llm()

        elif kind == "anthropic":
            dlg = AnthropicDialog(self)
            if dlg.exec() == QDialog.DialogCode.Accepted:
                key, model, remember = dlg.values()
                if not key:
                    QMessageBox.warning(self, "Missing API key", "Please enter a valid Anthropic API key.")
                    self.cbLLM.setCurrentIndex(0); return
                if not model:
                    QMessageBox.warning(self, "Missing model", "Please select or type a model name.")
                    self.cbLLM.setCurrentIndex(0); return
                os.environ["ANTHROPIC_API_KEY"] = key
                os.environ["ANTHROPIC_MODEL"] = model
                if remember:
                    cfg = load_config()
                    cfg["ANTHROPIC_API_KEY"] = key
                    cfg["ANTHROPIC_MODEL"] = model
                    save_config(cfg)
                try:
                    self.llm = AnthropicChat(model=model)
                    self.llm_backend = "anthropic"
                    QMessageBox.information(self, "LLM", f"Applied: {self.llm.name}")
                except Exception as e:
                    QMessageBox.warning(self, "LLM error", str(e))
                    self.cbLLM.setCurrentIndex(0)
            else:
                self.cbLLM.setCurrentIndex(0)
                self.apply_llm()

        else:
            self.apply_llm()

    # ---------- Indexing ----------
    def pick_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select a folder to index")
        if d:
            self.eFolder.setText(d)
            if not self.eIndexName.text().strip():
                self.eIndexName.setText(Path(d).name)

    def start_index(self):
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "Please wait", "Indexing in progress…"); return
        folder = self.eFolder.text().strip()
        if not folder:
            QMessageBox.warning(self, "Attention", "Please choose a folder."); return
        index_name = self.eIndexName.text().strip()
        if not index_name:
            QMessageBox.warning(self, "Attention", "Please enter an index name."); return
        index_dir = indexes_base_dir() / index_name
        cfg = IndexConfig(embed_model=self.cbEmbed.currentData(), index_type=self.cbIndex.currentData())
        self.worker = IndexThread(Path(folder), index_dir, cfg)
        self.worker.status.connect(self.log.append); self.worker.progress.connect(self.on_progress_update); self.worker.done.connect(self.index_done)
        self.bIndex.setEnabled(False); self.bCancel.setEnabled(True)
        self.log.append(f"\n— Starting indexing [{index_name}] → {folder}"); self.prog.setRange(0,0); self.worker.start()

    def on_progress_update(self, p:int):
        if self.prog.minimum()==0 and self.prog.maximum()==0: self.prog.setRange(0,100)
        self.prog.setValue(p)

    def cancel_index(self):
        if self.worker: self.worker.cancel()

    def index_done(self, files:int, vecs:int):
        self.bIndex.setEnabled(True); self.bCancel.setEnabled(False); self.prog.setRange(0,100); self.prog.setValue(100)
        name = self.eIndexName.text().strip()
        self.log.append(f"\n✅ Completed [{name}]. Files indexed: {files} • Vectors: {vecs}")
        self.retriever = None
        self._refresh_index_combo(select=name)

    # ---------- Chat ----------
    def _refresh_index_combo(self, select: str = "") -> None:
        current = select or self.cbIndexSel.currentText()
        self.cbIndexSel.clear()
        for name in list_indexes():
            self.cbIndexSel.addItem(name)
        if current and self.cbIndexSel.findText(current) >= 0:
            self.cbIndexSel.setCurrentText(current)

    def reload_index(self):
        name = self.cbIndexSel.currentText()
        if not name:
            QMessageBox.warning(self, "No index", "No indexes available. Build one first."); return
        index_dir = indexes_base_dir() / name
        try:
            self.retriever = Retriever(index_dir)
            self.lblIndex.setText(f"Loaded: {name}  • vectors={self.retriever.idx.ntotal}  • embed={self.retriever.embed_model}")
            QMessageBox.information(self, "OK", f"Index '{name}' loaded.")
        except Exception as e:
            self.lblIndex.setText(f"Error loading '{name}'")
            QMessageBox.warning(self, "Error", f"Could not load index '{name}'.\n{e}")

    def apply_llm(self):
        kind = self.cbLLM.currentData(); name = self.eModelName.text().strip() or ""
        try:
            if kind=="none":
                self.llm = NoLLM(); self.llm_backend = "none"
            elif kind=="openai":
                model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                self.llm = OpenAIChat(model=model); self.llm_backend = "openai"
            elif kind=="anthropic":
                model = os.getenv("ANTHROPIC_MODEL", ANTHROPIC_PRESETS[0])
                self.llm = AnthropicChat(model=model); self.llm_backend = "anthropic"
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
            name = self.cbIndexSel.currentText()
            if not name:
                QMessageBox.warning(self, "Missing index", "No index selected. Build or load one first."); return
            try: self.retriever = Retriever(indexes_base_dir() / name)
            except Exception:
                QMessageBox.warning(self, "Missing index", f"Could not load index '{name}'."); return
        if self.asker and self.asker.isRunning(): return
        self._current_q = q
        self.bAsk.setEnabled(False)
        self._render_history(f"<i>Searching for: \"{q}\"…</i>")
        self.asker = AskThread(q, self.retriever, self.llm); self.asker.ready.connect(self.on_answer_ready); self.asker.error.connect(self.on_answer_error); self.asker.start()

    def on_answer_ready(self, html: str):
        self._history.append((self._current_q, html))
        self._render_history()
        self.bAsk.setEnabled(True); self.inp.clear(); self.inp.setFocus()

    def on_answer_error(self, msg: str):
        QMessageBox.warning(self, "Error while answering", msg); self.bAsk.setEnabled(True); self.inp.setFocus()

    def _render_history(self, pending: str = "") -> None:
        parts = []
        for q, a in self._history:
            parts.append(
                f"<div style='background:#f0f4ff;padding:6px 8px;margin-bottom:4px;border-radius:4px'>"
                f"<b>Q:</b> {q}</div>"
                f"<div style='padding:4px 8px'>{a}</div>"
                f"<hr style='border:1px solid #ddd;margin:8px 0'>"
            )
        if pending:
            parts.append(f"<div style='color:gray'>{pending}</div>")
        self.out.setHtml("".join(parts))
        sb = self.out.verticalScrollBar()
        sb.setValue(sb.maximum())

    def clear_history(self) -> None:
        self._history.clear()
        self.out.clear()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    icon = _app_icon()
    app.setWindowIcon(icon)
    w = App()
    w.setWindowIcon(icon)
    w.show()
    sys.exit(app.exec())
