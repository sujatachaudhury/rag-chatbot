import streamlit as st

from src.agent import ask_question, get_retriever
from src.config import PDF_DIR
from src.vectorstore import ingest_pdf_file

st.set_page_config(page_title="YTRAG", page_icon="📄")
st.title("Agentic RAG over your PDFs")
st.caption("Upload a PDF to add it to the corpus, then ask questions grounded in everything ingested so far.")

retriever = get_retriever()

with st.sidebar:
    st.header("Add a document")
    uploaded = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded and st.button("Add to corpus"):
        PDF_DIR.mkdir(parents=True, exist_ok=True)
        dest = PDF_DIR / uploaded.name
        dest.write_bytes(uploaded.getvalue())
        with st.spinner(f"Chunking and embedding {uploaded.name}..."):
            n_chunks = ingest_pdf_file(
                dest, embedder=retriever.embedding_manager, store=retriever.vector_store
            )
        st.success(f"Added {n_chunks} chunks from {uploaded.name}.")

    st.caption(f"Corpus size: {retriever.vector_store.count()} chunks")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about the ingested PDFs")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = ask_question(question)
        st.markdown(result["answer"])
        if result["grounded"] and result["documents"]:
            with st.expander("Sources"):
                for doc in result["documents"]:
                    source = doc["metadata"].get("source_file", "unknown")
                    st.caption(f"{source} · similarity {doc['similarity_score']:.2f}")
                    st.text(doc["content"][:300] + "...")

    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
