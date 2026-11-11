
import streamlit as st
import os
import hashlib

from pdf_qa import (
    process_pdf,
    retrieve_chunks,
    generate_with_retrieval
)


# 1. Page configuration

st.set_page_config(
    page_title="PDF LLM Assistant",
    page_icon="📚",
    layout="wide"
)


# 2. Application title

st.title("📚 PDF Question Answering Assistant")

st.caption(
    "Upload a PDF, ask questions and get AI-powered answers using Ollama."
)


# 3. Sidebar

st.sidebar.title("⚙️ Settings")

st.sidebar.markdown("### PDF Configuration")

chunk_size = st.sidebar.selectbox(
    "Chunk Size",
    [300, 500, 800, 1000],
    index=1
)

chunk_overlap = st.sidebar.selectbox(
    "Chunk Overlap",
    [50, 100, 150, 200],
    index=1
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Model: llama3.2\n\n"
    "Database: ChromaDB\n\n"
    "Retrieval: Top-5"
)


# 4. PDF upload

uploaded_file = st.file_uploader(
    "Upload your PDF document",
    type=["pdf"]
)


# 5. Initialize session state

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "collection" not in st.session_state:
    st.session_state.collection = None

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

if "pdf_hash" not in st.session_state:
    st.session_state.pdf_hash = None


# 6. Process uploaded PDF

if uploaded_file is not None:

    pdf_bytes = uploaded_file.getvalue()

    current_hash = hashlib.sha256(pdf_bytes).hexdigest()

    if current_hash != st.session_state.pdf_hash:

        os.makedirs("dataset/uploaded_files", exist_ok=True)

        pdf_path = os.path.join(
            "dataset",
            "uploaded_files",
            uploaded_file.name
        )

        with open(pdf_path, "wb") as file:
            file.write(pdf_bytes)

        collection_name = "pdf_" + current_hash[:16]

        try:

            with st.spinner("Extracting PDF text and creating ChromaDB collection..."):

                result = process_pdf(
                    pdf_path,
                    collection_name,
                    chunk_size,
                    chunk_overlap
                )

            st.session_state.collection = result["collection"]

            st.session_state.pdf_hash = current_hash

            st.session_state.pdf_processed = True

            st.session_state.pdf_name = uploaded_file.name

            st.session_state.total_pages = result["pages"]

            st.session_state.total_chunks = len(result["chunks"])

            st.session_state.chat_history = []

            st.success("PDF processed successfully!")

        except Exception as e:

            st.error(f"PDF processing failed: {e}")

            st.session_state.pdf_processed = False


# 7. Display PDF information

if st.session_state.pdf_processed:

    st.markdown("### 📄 Uploaded PDF Information")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "PDF Filename",
        st.session_state.pdf_name
    )

    col2.metric(
        "Total Pages",
        st.session_state.total_pages
    )

    col3.metric(
        "Total Chunks",
        st.session_state.total_chunks
    )

    st.markdown("---")


# 8. Display previous chat messages

for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# 9. Question input

if st.session_state.pdf_processed:

    question = st.chat_input(
        "Ask a question about your PDF..."
    )

    if question:

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner("Searching PDF and generating answer..."):

                try:

                    retrieved_chunks = retrieve_chunks(
                        st.session_state.collection,
                        question
                    )

                    answer = generate_with_retrieval(
                        question,
                        retrieved_chunks
                    )

                    st.markdown(answer)

                    with st.expander("View Top-5 Retrieved Chunks"):

                        for index, chunk in enumerate(
                            retrieved_chunks,
                            start=1
                        ):

                            st.markdown(
                                f"**Chunk {index}**"
                            )

                            st.write(chunk)

                            st.markdown("---")

                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    )

                except Exception as e:

                    st.error(f"Error generating answer: {e}")

else:

    st.info("Please upload a PDF document to start asking questions.")


# 10. Clear chat history

if st.sidebar.button("🗑️ Clear Chat History"):

    st.session_state.chat_history = []

    st.rerun()
