import os
import re

import faiss
import numpy as np

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# ============================================================
# EMBEDDING MODEL
# ============================================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ============================================================
# DOCUMENT LOADER
# ============================================================

def load_pdf(
    file_path,
    department
):

    reader = PdfReader(
        file_path
    )

    documents = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text()

        if not text:
            continue

        text = re.sub(
            r"\s+",
            " ",
            text
        ).strip()

        if text:

            documents.append(
                {
                    "text": text,
                    "source": os.path.basename(
                        file_path
                    ),
                    "page": page_number,
                    "department": department
                }
            )

    return documents
# ============================================================
# LOAD ALL DOCUMENTS
# ============================================================

def load_documents(
    base_folder="documents"
):

    all_documents = []

    departments = [
        "hr",
        "technical",
        "projects"
    ]

    for department in departments:

        folder = os.path.join(
            base_folder,
            department
        )

        if not os.path.exists(folder):

            continue

        for filename in os.listdir(folder):

            if not filename.lower().endswith(
                ".pdf"
            ):

                continue

            path = os.path.join(
                folder,
                filename
            )

            documents = load_pdf(
                path,
                department
            )

            all_documents.extend(
                documents
            )

    return all_documents


# ============================================================
# CHUNK DOCUMENTS
# ============================================================

def create_chunks(
    documents,
    chunk_size=700,
    overlap=100
):

    chunks = []

    metadata = []

    for document in documents:

        text = document["text"]

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk = text[start:end]

            if chunk.strip():

                chunks.append(
                    chunk.strip()
                )

                metadata.append(
                    {
                        "source":
                            document["source"],

                        "page":
                            document["page"],

                        "department":
                            document["department"]
                    }
                )

            start += (
                chunk_size - overlap
            )

    return chunks, metadata


# ============================================================
# BUILD VECTOR INDEX
# ============================================================

def build_index(
    chunks
):

    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = embeddings.astype(
        "float32"
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    return index


# ============================================================
# RETRIEVE DOCUMENTS
# ============================================================

def retrieve(
    question,
    chunks,
    metadata,
    index,
    department=None,
    top_k=5
):

    question_embedding = (
        embedding_model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
    )

    question_embedding = (
        question_embedding.astype(
            "float32"
        )
    )

    scores, indices = index.search(
        question_embedding,
        min(top_k * 3, len(chunks))
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx < 0:
            continue

        item_metadata = metadata[idx]

        if (
            department
            and item_metadata["department"]
            != department
        ):

            continue

        results.append(
            {
                "text":
                    chunks[idx],

                "metadata":
                    item_metadata,

                "score":
                    float(score)
            }
        )

        if len(results) >= top_k:

            break

    return results