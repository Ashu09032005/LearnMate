import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

from config import GROQ_MODEL, TEMPERATURE, CHUNK_SIZE, CHUNK_OVERLAP


# Fast embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def process_pdfs(uploaded_files):
    """
    Takes list of uploaded PDFs and returns FAISS vectorstore
    """

    all_documents = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        loader = PyPDFLoader(temp_path)
        documents = loader.load()

        # Add filename metadata
        for doc in documents:
            doc.metadata["source_file"] = uploaded_file.name

        all_documents.extend(documents)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(all_documents)

    vectorstore = FAISS.from_documents(chunks, embedding_model)

    return vectorstore


def build_qa_chain(vectorstore):
    """
    Builds RetrievalQA chain using Groq
    """

    retriever = vectorstore.as_retriever()

    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=TEMPERATURE
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    return qa_chain


def summarize_documents(vectorstore):
    """
    Generates summary for all uploaded documents
    """

    retriever = vectorstore.as_retriever()

    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=TEMPERATURE
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    response = qa_chain.invoke(
        "Provide a detailed summary of all the uploaded documents."
    )

    return response["result"]
