import streamlit as st

from rag import load_documents, create_chunks, build_index
from graph import create_graph


st.set_page_config(
    page_title="Enterprise AI Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("AI Enterprise Knowledge Assistant")
st.write("Ask questions about HR policies, technical documentation and company projects.")

if "initialized" not in st.session_state:
    st.session_state.initialized = False

if "messages" not in st.session_state:
    st.session_state.messages = []


@st.cache_resource
def initialize_knowledge_base():
    documents = load_documents("documents")
    if not documents:
        raise ValueError("No PDF documents found.")
    chunks, metadata = create_chunks(documents)
    index = build_index(chunks)
    graph = create_graph(chunks, metadata, index)
    return documents, chunks, metadata, index, graph


with st.sidebar:
    st.header("Enterprise Knowledge Base")

    if st.button("Load Documents", use_container_width=True):
        try:
            with st.spinner("Building knowledge base..."):
                documents, chunks, metadata, index, graph = initialize_knowledge_base()
                st.session_state.documents = documents
                st.session_state.chunks = chunks
                st.session_state.metadata = metadata
                st.session_state.index = index
                st.session_state.graph = graph
                st.session_state.initialized = True
                st.success(f"Loaded {len(documents)} pages.")
        except Exception as e:
            st.error(f"Initialization error: {e}")

    st.divider()
    st.subheader("Departments")
    st.write("HR")
    st.write("Technical")
    st.write("Projects")
    st.divider()

    if st.session_state.initialized:
        st.success(f"{len(st.session_state.chunks)} chunks indexed")


if not st.session_state.initialized:
    try:
        with st.spinner("Loading enterprise knowledge base..."):
            documents, chunks, metadata, index, graph = initialize_knowledge_base()
            st.session_state.documents = documents
            st.session_state.chunks = chunks
            st.session_state.metadata = metadata
            st.session_state.index = index
            st.session_state.graph = graph
            st.session_state.initialized = True
    except Exception as e:
        st.warning("Please add PDF documents to the documents folder.")
        st.info(f"Details: {e}")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


question = st.chat_input("Ask about company policies, technology or projects...")

if question:
    if not st.session_state.initialized:
        st.error("Knowledge base is not initialized.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("AI agents are working..."):
            try:
                result = st.session_state.graph.invoke({
                    "question": question,
                    "category": "",
                    "answer": "",
                    "context": "",
                    "sources": []
                })

                answer = result["answer"]
                category = result["category"]
                sources = result["sources"]

                st.caption(f"Agent selected: {category}")
                st.markdown(answer)

                if sources:
                    with st.expander("View Sources"):
                        seen = set()
                        for source in sources:
                            key = (source["source"], source["page"])
                            if key in seen:
                                continue
                            seen.add(key)
                            st.write(f"• {source['source']} — Page {source['page']}")

                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(f"Error: {e}")
