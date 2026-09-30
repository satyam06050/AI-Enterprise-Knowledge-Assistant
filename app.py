import hashlib

import streamlit as st

from graph import UPLOADED_NOT_FOUND_MESSAGE, create_graph, create_uploaded_graph
from rag import build_index, create_chunks, ingest_uploaded_document, load_documents


st.set_page_config(page_title="Enterprise Knowledge Assistant", layout="wide")

st.markdown(
    """
    <style>
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background: #F5F3EC; color: #18181B; }
        [data-testid="stHeader"], [data-testid="stToolbar"] { background: #F5F3EC; }
        [data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child { background: #FFFFFF; }
        [data-testid="stSidebar"] { border-right: 1px solid #DED9D0; }
        [data-testid="stSidebar"] * { color: #18181B; }
        .block-container { max-width: 1120px; padding-top: 2.75rem; padding-bottom: 3rem; }
        h1, h2, h3, p, label, .stMarkdown { color: #18181B; }
        .eyebrow { color: #7E5281; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; }
        .intro { max-width: 680px; color: #5F5A63; font-size: 1.03rem; }
        .stTabs [data-baseweb="tab-list"] { gap: 1.75rem; border-bottom: 1px solid #D9D4CA; }
        .stTabs [data-baseweb="tab"] { color: #5F5A63; font-weight: 650; padding: 0.75rem 0; }
        .stTabs [aria-selected="true"] { color: #7E5281; }
        .stTabs [data-baseweb="tab-highlight"] { background-color: #7E5281; }
        .stTextInput input, .stTextArea textarea, .stSelectbox [data-baseweb="select"] > div, .stFileUploader { background: #FFFFFF; border-color: #D8D2C8; color: #18181B !important; }
        .stTextInput input::placeholder, .stTextArea textarea::placeholder { color: #746E76 !important; opacity: 1; }
        .stTextInput input:focus, .stTextArea textarea:focus { border-color: #7E5281; box-shadow: 0 0 0 1px #7E5281; }
        [data-baseweb="select"] input, [data-baseweb="select"] span { color: #18181B !important; }
        .stButton > button { background: #7E5281; border: 1px solid #7E5281; color: #FFFFFF; border-radius: 6px; font-weight: 650; }
        .stButton > button:hover { background: #68436B; border-color: #68436B; color: #FFFFFF; }
        .document-row { background: #FFFFFF; border: 1px solid #DED9D0; border-radius: 6px; padding: 0.85rem 1rem; margin-bottom: 0.55rem; }
        .document-name { color: #18181B; font-weight: 650; }
        .document-meta { color: #6B6670; font-size: 0.86rem; margin-top: 0.16rem; }
        .status-ready { color: #456450; font-weight: 650; }
    </style>
    """,
    unsafe_allow_html=True,
)

for key, default in (
    ("initialized", False),
    ("uploaded_documents", []),
    ("uploaded_chunks", []),
    ("uploaded_metadata", []),
    ("uploaded_index", None),
):
    if key not in st.session_state:
        st.session_state[key] = default


@st.cache_resource
def initialize_knowledge_base():
    documents = load_documents("documents")
    if not documents:
        raise ValueError("No company documents were found.")
    chunks, metadata = create_chunks(documents)
    index = build_index(chunks)
    graph = create_graph(chunks, metadata, index)
    return documents, chunks, metadata, index, graph


def initialize_company_knowledge():
    documents, chunks, metadata, index, graph = initialize_knowledge_base()
    st.session_state.documents = documents
    st.session_state.chunks = chunks
    st.session_state.metadata = metadata
    st.session_state.index = index
    st.session_state.graph = graph
    st.session_state.initialized = True


def source_label(source):
    if source.get("page"):
        return f"{source['source']} - page {source['page']}"
    if source.get("row"):
        return f"{source['source']} - row {source['row']}"
    return source["source"]


st.markdown('<div class="eyebrow">Private knowledge workspace</div>', unsafe_allow_html=True)
st.title("Enterprise Knowledge Assistant")
st.markdown(
    '<p class="intro">Ask the company knowledge base, or upload a document for a focused session-only review.</p>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Company knowledge")
    if st.button("Refresh company documents", use_container_width=True):
        try:
            with st.spinner("Building company knowledge base..."):
                initialize_company_knowledge()
            st.success("Company knowledge is ready.")
        except Exception as error:
            st.error(f"Initialization error: {error}")

    if st.session_state.initialized:
        st.caption(f"{len(st.session_state.chunks)} company chunks indexed")


if not st.session_state.initialized:
    try:
        with st.spinner("Loading company knowledge base..."):
            initialize_company_knowledge()
    except Exception as error:
        st.warning("Company knowledge could not be initialized.")
        st.caption(str(error))


company_tab, upload_tab = st.tabs(["Company Knowledge", "Uploaded Document"])

with company_tab:
    company_question = st.text_input(
        "Ask a company question",
        placeholder="Ask about HR policies, technology, or company projects.",
        key="company_question",
    )
    if st.button("Get company answer", key="company_answer"):
        if not st.session_state.initialized:
            st.error("Company knowledge is not initialized.")
        elif not company_question.strip():
            st.warning("Enter a question to continue.")
        else:
            with st.spinner("AI agents are working..."):
                try:
                    result = st.session_state.graph.invoke({
                        "question": company_question.strip(),
                        "category": "",
                        "answer": "",
                        "context": "",
                        "sources": [],
                    })
                    st.caption(f"Agent selected: {result['category']}")
                    st.markdown(result["answer"])
                    if result["sources"]:
                        with st.expander("Sources"):
                            for source in {source_label(item) for item in result["sources"]}:
                                st.write(source)
                except Exception as error:
                    st.error(f"Unable to answer the company question: {error}")

with upload_tab:
    upload_column, status_column = st.columns([1.05, 0.95], gap="large")
    with upload_column:
        uploaded_file = st.file_uploader(
            "Upload document",
            type=["pdf", "txt", "docx", "csv"],
            help="Uploaded files are retained only for the current Streamlit session.",
        )
        if st.button("Process document", key="process_document"):
            if uploaded_file is None:
                st.warning("Choose a PDF, TXT, DOCX, or CSV file first.")
            else:
                document_id = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
                existing = next(
                    (item for item in st.session_state.uploaded_documents if item["id"] == document_id),
                    None,
                )
                if existing:
                    st.info(f"{existing['filename']} is already ready for questions.")
                else:
                    with st.spinner("Extracting, chunking, embedding, and indexing..."):
                        try:
                            processed = ingest_uploaded_document(uploaded_file, document_id)
                            if st.session_state.uploaded_index is None:
                                st.session_state.uploaded_index = build_index(processed["chunks"])
                            else:
                                new_index = build_index(processed["chunks"])
                                st.session_state.uploaded_index.add(new_index.reconstruct_n(0, new_index.ntotal))

                            st.session_state.uploaded_chunks.extend(processed["chunks"])
                            st.session_state.uploaded_metadata.extend(processed["metadata"])
                            st.session_state.uploaded_documents.append({
                                "id": processed["document_id"],
                                "filename": processed["filename"],
                                "file_type": processed["file_type"],
                                "source_units": processed["source_units"],
                                "chunks": len(processed["chunks"]),
                                "status": "Ready",
                            })
                            st.success(f"{processed['filename']} is ready for questions.")
                        except Exception as error:
                            st.error(f"Could not process the document: {error}")

    with status_column:
        st.subheader("Session documents")
        if not st.session_state.uploaded_documents:
            st.caption("No documents have been processed in this session.")
        else:
            for item in st.session_state.uploaded_documents:
                st.markdown(
                    f'''<div class="document-row">
                        <div class="document-name">{item["filename"]}</div>
                        <div class="document-meta">{item["file_type"]} | {item["source_units"]} source sections | {item["chunks"]} chunks</div>
                        <div class="document-meta status-ready">{item["status"]}</div>
                    </div>''',
                    unsafe_allow_html=True,
                )

    if st.session_state.uploaded_documents:
        st.divider()
        selected_document_id = st.selectbox(
            "Document for this question",
            options=[item["id"] for item in st.session_state.uploaded_documents],
            format_func=lambda item_id: next(
                item["filename"] for item in st.session_state.uploaded_documents if item["id"] == item_id
            ),
        )
        uploaded_question = st.text_input(
            "Ask about the selected document",
            placeholder="Ask a question grounded in this document.",
            key="uploaded_question",
        )
        if st.button("Ask document", key="ask_document"):
            if not uploaded_question.strip():
                st.warning("Enter a question to continue.")
            else:
                with st.spinner("Searching the selected document..."):
                    try:
                        upload_graph = create_uploaded_graph(
                            st.session_state.uploaded_chunks,
                            st.session_state.uploaded_metadata,
                            st.session_state.uploaded_index,
                            selected_document_id,
                        )
                        result = upload_graph.invoke({
                            "question": uploaded_question.strip(),
                            "category": "UPLOADED",
                            "answer": "",
                            "context": "",
                            "sources": [],
                        })
                        st.markdown(result["answer"])
                        if result["sources"] and result["answer"] != UPLOADED_NOT_FOUND_MESSAGE:
                            st.markdown("**Sources**")
                            for source in {source_label(item) for item in result["sources"]}:
                                st.markdown(f"- {source}")
                    except Exception as error:
                        st.error(f"Unable to answer the document question: {error}")
