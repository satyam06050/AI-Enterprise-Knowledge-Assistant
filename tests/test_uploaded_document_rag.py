import os

os.environ.setdefault("GROQ_API_KEY", "test-key")

import graph
from graph import create_uploaded_graph
from rag import build_index, ingest_uploaded_document


class UploadedFile:
    def __init__(self, name, content):
        self.name = name
        self._content = content

    def getvalue(self):
        return self._content


def test_uploaded_document_flow_returns_answer_and_citation(monkeypatch):
    uploaded_file = UploadedFile(
        "release_notes.txt",
        b"The Orion release is scheduled for 14 October. The deployment owner is Maya.",
    )

    processed = ingest_uploaded_document(uploaded_file, document_id="release-notes")
    index = build_index(processed["chunks"])

    monkeypatch.setattr(
        graph,
        "generate_uploaded_answer",
        lambda question, context: "The Orion release is scheduled for 14 October.",
    )
    upload_graph = create_uploaded_graph(
        processed["chunks"],
        processed["metadata"],
        index,
        processed["document_id"],
    )
    result = upload_graph.invoke({
        "question": "When is the Orion release scheduled?",
        "category": "UPLOADED",
        "answer": "",
        "context": "",
        "sources": [],
    })

    assert result["answer"] == "The Orion release is scheduled for 14 October."
    assert result["sources"][0]["source"] == "release_notes.txt"
    assert result["sources"][0]["document_id"] == "release-notes"


def test_uploaded_csv_retains_row_metadata():
    uploaded_file = UploadedFile(
        "contacts.csv",
        b"name,role\nMaya,Deployment owner\n",
    )

    processed = ingest_uploaded_document(uploaded_file, document_id="contacts")

    assert processed["metadata"][0]["source"] == "contacts.csv"
    assert processed["metadata"][0]["row"] == 2
