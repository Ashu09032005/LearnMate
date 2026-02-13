import streamlit as st
from rag_pipeline import process_pdfs, build_qa_chain, summarize_documents

st.set_page_config(page_title="PDF RAG Chatbot", layout="wide")
st.title("LearnMate")

uploaded_files = st.file_uploader(
    "Upload one or more PDFs",
    type="pdf",
    accept_multiple_files=True
)

# Process PDFs
if uploaded_files:

    if "vectorstore" not in st.session_state:
        with st.spinner("Processing PDFs..."):
            vectorstore = process_pdfs(uploaded_files)
            st.session_state.vectorstore = vectorstore
        st.success("PDFs processed successfully!")

# Ask questions
if "vectorstore" in st.session_state:

    qa_chain = build_qa_chain(st.session_state.vectorstore)

    query = st.text_input("Ask a question from the documents")

    if query:
        with st.spinner("Generating answer..."):
            response = qa_chain.invoke(query)

        st.write("### Answer:")
        st.write(response["result"])

        with st.expander("Sources"):
            for doc in response["source_documents"]:
                st.write(f"Source: {doc.metadata.get('source_file')}")
                st.write(doc.page_content[:300] + "...")
                st.write("---")

    # Summary Button
    if st.button("Summarize All PDFs"):
        with st.spinner("Generating summary..."):
            summary = summarize_documents(st.session_state.vectorstore)

        st.write("### 📄 Summary:")
        st.write(summary)
