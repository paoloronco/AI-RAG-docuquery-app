# RAG-App 🧠📄  

An **AI-powered document search application** based on **RAG (Retrieval-Augmented Generation)**.  
The app lets you **index your local documents** using FAISS and embedding models, and then **query them in natural language** through a simple **PyQt6 GUI**.  
It’s like having a **personal AI assistant** trained only on your files.  

---

## 🚀 Features  
- Index local documents with **FAISS**  
- Use **SentenceTransformers** for embeddings  
- Query in **natural language**  
- GPU acceleration (CUDA) support when available  
- Simple GUI built with **PyQt6**  

---

## 📌 Version 1.0  

This is the **first stable version** of the project.  
- Basic document indexing with **FAISS** is working  
- Queries in **natural language** can be executed against indexed documents  
- Support for both **CPU and GPU (CUDA)**  
- Simple **PyQt6 GUI** available for interaction  

### ⚠️ Current Limitations  
- Indexing is **not stable on large datasets**: performance may degrade when too many documents are ingested at once  
- The **hallucination issue** still occurs: in some cases, when searching for non-existent information, the system may return an unrelated snippet from the documents  
- Indexing can only be run via **Python scripts**, not through the GUI executable  
- The GUI may **occasionally crash** during use  

---

## 📂 Project Structure  
```bash
Project/
│── .venv/ # Virtual environment (ignored in git)
│── faiss_index/ # Local FAISS index (ignored in git)
│ ├── index.faiss
│ └── meta.jsonl
│── pipeline_fais/ # FAISS-based pipeline
│ ├── index_faiss.py
│ ├── main_faiss.py
│ ├── query_faiss.py
│ └── requirements.txt
│── check_cuda.py # Utility script to check CUDA availability
│── .gitignore # Git ignore rules
│── readme.md # Project description
```
---

## ⚙️ Installation  

> Requires **Python 3.12** (64-bit).

Clone the repo:  
```bash
git clone https://github.com/paoloronco/AI-RAG-docuquery-app.git
cd AI-RAG-docuquery-app
```
Create and activate virtual environment:
```bash
py -3.12 -m venv .venv
.venv\Scripts\activate      # On Windows
source .venv/bin/activate   # On Linux/Mac
```

Install dependencies:
```bash
pip install -r requirements.txt
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r pipeline_faiss\requirements.txt
python -c "import faiss,sys; print('OK:', faiss.__file__); print(sys.executable)"
```

### ▶️ Usage (from source)

Index your documents:
```bash
python pipeline_faiss\index_faiss.py "C:\path\to\your\docs"
```

Start the app with GUI:
```bash
python pipeline_faiss\main_faiss.py
```

Check if CUDA is available:
```bash
python check_cuda.py
```

### 📦 Build Executable (Optional)
If you want a standalone .exe:
```bash
pip install pyinstaller
pyinstaller pipeline_faiss\main_faiss.py --name RAG-App --onefile --noconsole
```

---

## 📝 License

This project is released under the MIT License. Feel free to use and modify it.

## 🙌 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.