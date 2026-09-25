import os
import tempfile

from rag import build_index, create_chunks, load_documents, retrieve


def test_load_documents_supports_text_files(tmp_path):
    hr_dir = tmp_path / "hr"
    hr_dir.mkdir()
    (hr_dir / "leave_policy.txt").write_text("Annual leave is 20 days per year.", encoding="utf-8")

    docs = load_documents(base_folder=str(tmp_path))

    assert len(docs) == 1
    assert docs[0]["department"] == "hr"
    assert "leave" in docs[0]["text"].lower()


def test_retrieve_returns_relevant_chunk(tmp_path):
    technical_dir = tmp_path / "technical"
    technical_dir.mkdir()
    (technical_dir / "deploy.txt").write_text(
        "To deploy the service, run the build script and then restart the application.",
        encoding="utf-8",
    )

    documents = load_documents(base_folder=str(tmp_path))
    chunks, metadata = create_chunks(documents, chunk_size=200, overlap=20)
    index = build_index(chunks)

    result = retrieve("How do I deploy the service?", chunks, metadata, index, top_k=1)

    assert result
    assert "deploy" in result[0]["text"].lower()
