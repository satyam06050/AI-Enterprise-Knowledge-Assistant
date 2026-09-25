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