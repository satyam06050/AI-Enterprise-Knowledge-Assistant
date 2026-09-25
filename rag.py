import os
import re

import faiss
import numpy as np

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


def get_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def load_pdf(file_path, department):
    reader = PdfReader(file_path)
    documents = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if not text:
            continue
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            documents.append({
                "text": text,
                "source": os.path.basename(file_path),
                "page": page_number,
                "department": department
            })
    return documents


def load_txt(file_path, department):
    with open(file_path, encoding="utf-8") as f:
        text = re.sub(r"\s+", " ", f.read()).strip()
    if not text:
        return []
    return [{
        "text": text,
        "source": os.path.basename(file_path),
        "page": 1,
        "department": department
    }]


def load_documents(base_folder="documents"):
    all_documents = []
    for department in ["hr", "technical", "projects"]:
        folder = os.path.join(base_folder, department)
        if not os.path.exists(folder):
            continue
        for filename in os.listdir(folder):
            path = os.path.join(folder, filename)
            lower = filename.lower()
            if lower.endswith(".pdf"):
                all_documents.extend(load_pdf(path, department))
            elif lower.endswith(".txt"):
                all_documents.extend(load_txt(path, department))
    return all_documents


def create_chunks(documents, chunk_size=700, overlap=100):
    chunks = []
    metadata = []
    for document in documents:
        text = document["text"]
        start = 0
        while start < len(text):
            chunk = text[start:start + chunk_size]
            if chunk.strip():
                chunks.append(chunk.strip())
                metadata.append({
                    "source": document["source"],
                    "page": document["page"],
                    "department": document["department"]
                })
            start += chunk_size - overlap
    return chunks, metadata


def build_index(chunks, model):
    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def retrieve(question, chunks, metadata, index, model, department=None, top_k=5):
    q_emb = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(q_emb, min(top_k * 3, len(chunks)))

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        m = metadata[idx]
        if department and m["department"] != department:
            continue
        results.append({
            "text": chunks[idx],
            "metadata": m,
            "score": float(score)
        })
        if len(results) >= top_k:
            break
    return results
