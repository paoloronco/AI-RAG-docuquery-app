from dataclasses import dataclass
from typing import Literal

EMBED_MODELS = {
    "intfloat/e5-small-v2": {"display": "e5-small-v2 (fast, CPU)"},
    "intfloat/e5-base-v2": {"display": "e5-base-v2 (medium quality)"},
    "intfloat/e5-large-v2": {"display": "e5-large-v2 (high quality, GPU recommended)"},
    "sentence-transformers/all-MiniLM-L6-v2": {"display": "MiniLM-L6-v2 (very fast)"},
}

INDEX_TYPES = [
    ("flat", "FAISS Flat (precise, RAM)"),
    ("hnsw", "FAISS HNSW (ANN, scales well)"),
    ("ivf",  "FAISS IVF Flat (ANN, large datasets)"),
]

LLM_BACKENDS = [
    ("none", "No LLM (citations only)"),
    ("openai", "OpenAI / compatible"),
    ("hf_local", "Local HuggingFace"),
]

DEFAULTS = {
    "embed_model": "intfloat/e5-small-v2",
    "index_type": "hnsw",
    "bm25": True,
    "k": 12,
    "rerank": False,
    "rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "min_sim": 0.25,
}

@dataclass
class IndexConfig:
    embed_model: str
    index_type: Literal["flat", "hnsw", "ivf"]
    created_by: str = "rag_pro"
