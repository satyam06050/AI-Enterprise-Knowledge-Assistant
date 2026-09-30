import os
import re
import csv
import io
import uuid
import zipfile
from xml.etree import ElementTree

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

            path = os.path.join(folder, filename)
            lower = filename.lower()

            if lower.endswith(".pdf"):
                documents = load_pdf(path, department)
            elif lower.endswith(".txt"):
                documents = load_txt(path, department)
            else:
                continue

            all_documents.extend(documents)

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
                        key: value
                        for key, value in document.items()
                        if key != "text"
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
    document_id=None,
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

    candidate_count = min(
        len(chunks) if document_id else top_k * 3,
        len(chunks)
    )

    scores, indices = index.search(
        question_embedding,
        candidate_count
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
            and item_metadata.get("department")
            != department
        ):

            continue

        if (
            document_id
            and item_metadata.get("document_id")
            != document_id
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


# ============================================================
# SESSION UPLOADS
# ============================================================

def _clean_text(text):
    return re.sub(r"\s+", " ", text or "").strip()


def _uploaded_metadata(filename, document_id, **location):
    return {
        "source": filename,
        "department": "uploaded",
        "document_id": document_id,
        **location,
    }


def _extract_uploaded_pdf(file_bytes, filename, document_id):
    reader = PdfReader(io.BytesIO(file_bytes))
    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = _clean_text(page.extract_text())
        if text:
            documents.append({
                "text": text,
                **_uploaded_metadata(filename, document_id, page=page_number),
            })

    return documents


def _extract_uploaded_txt(file_bytes, filename, document_id):
    text = _clean_text(file_bytes.decode("utf-8-sig", errors="replace"))
    if not text:
        return []
    return [{
        "text": text,
        **_uploaded_metadata(filename, document_id),
    }]


def _extract_uploaded_docx(file_bytes, filename, document_id):
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as archive:
            xml = archive.read("word/document.xml")
    except (KeyError, zipfile.BadZipFile) as error:
        raise ValueError("The DOCX file could not be read.") from error

    root = ElementTree.fromstring(xml)
    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    paragraphs = [
        _clean_text("".join(paragraph.itertext()))
        for paragraph in root.iter(f"{namespace}p")
    ]
    text = " ".join(paragraph for paragraph in paragraphs if paragraph)

    if not text:
        return []
    return [{
        "text": text,
        **_uploaded_metadata(filename, document_id),
    }]


def _extract_uploaded_csv(file_bytes, filename, document_id):
    rows = list(csv.reader(io.StringIO(file_bytes.decode("utf-8-sig", errors="replace"))))
    if not rows:
        return []

    headers = rows[0]
    documents = []
    for row_number, row in enumerate(rows[1:], start=2):
        cells = [
            f"{headers[column]}: {value.strip()}"
            for column, value in enumerate(row)
            if column < len(headers) and value.strip()
        ]
        if cells:
            documents.append({
                "text": " ".join(cells),
                **_uploaded_metadata(filename, document_id, row=row_number),
            })

    return documents


def ingest_uploaded_document(uploaded_file, document_id=None):
    """Extract and chunk a Streamlit upload without saving it to documents/."""
    filename = uploaded_file.name
    extension = os.path.splitext(filename)[1].lower()
    extractors = {
        ".pdf": _extract_uploaded_pdf,
        ".txt": _extract_uploaded_txt,
        ".docx": _extract_uploaded_docx,
        ".csv": _extract_uploaded_csv,
    }

    if extension not in extractors:
        raise ValueError("Upload a PDF, TXT, DOCX, or CSV file.")

    file_bytes = uploaded_file.getvalue()
    if not file_bytes:
        raise ValueError("The uploaded file is empty.")

    resolved_document_id = document_id or str(uuid.uuid4())
    documents = extractors[extension](file_bytes, filename, resolved_document_id)
    if not documents:
        raise ValueError("No readable text was found in the uploaded document.")

    chunks, metadata = create_chunks(documents)
    for chunk_number, item_metadata in enumerate(metadata, start=1):
        item_metadata["chunk"] = chunk_number

    return {
        "document_id": resolved_document_id,
        "filename": filename,
        "file_type": extension.lstrip(".").upper(),
        "source_units": len(documents),
        "chunks": chunks,
        "metadata": metadata,
    }
