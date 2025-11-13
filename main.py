import streamlit as st
import os
import tempfile
import pandas as pd

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --------------------------
# API KEY
# --------------------------
api_key = st.secrets.get("GROQ_API_KEY")
if not api_key:
    st.error("❌ Add GROQ_API_KEY inside Streamlit Secrets.")
    st.stop()

# --------------------------
# UI Setup
# --------------------------
st.set_page_config(page_title="Smart File Assistant", layout="wide")
st.title("📘 Multi-File RAG System + Smart Recommendations")

uploaded_files = st.file_uploader(
    "Upload files: PDF / TXT / CSV",
    type=["pdf", "txt", "csv"],
    accept_multiple_files=True
)

# --------------------------
# LOAD FILES — FIXED & CLEAN
# --------------------------
def load_file(uploaded_file):
    suffix = uploaded_file.name.split(".")[-1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name

    # PDF
    if suffix == "pdf":
        loader = PyPDFLoader(path)
        docs = loader.load()

    # TXT
    elif suffix == "txt":
        loader = TextLoader(path)
        docs = loader.load()

    # CSV → convert whole table to text
    elif suffix == "csv":
        df = pd.read_csv(path)
        text = df.to_string()
        docs = [
            Document(
                page_content=text,
                metadata={"source_file": uploaded_file.name}
            )
        ]

    else:
        return []

    # Ensure every doc is LangChain Document
    cleaned = []
    for d in docs:
        if isinstance(d, Document):
            d.metadata["source_file"] = uploaded_file.name
            cleaned.append(d)
        else:
            # Convert dict to Document (if ever happens)
            cleaned.append(
                Document(
                    page_content=d["page_content"],
                    metadata=d.get("metadata", {"source_file": uploaded_file.name})
                )
            )

    return cleaned


# --------------------------
# BUILD RAG PIPELINE
# --------------------------
if uploaded_files:

    all_docs = []
    for f in uploaded_files:
        all_docs.extend(load_file(f))

    st.success(f"✔ Loaded {len(all_docs)} documents from {len(uploaded_files)} files.")

    # Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(all_docs)

    # Embeddings & Vector Store
    embedding = FastEmbedEmbeddings()
    vectorstore = Chroma.from_documents(chunks, embedding=embedding)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # LLM (Groq)
    llm = ChatOpenAI(
        model="llama-3.3-70b-versatile",
        openai_api_base="https://api.groq.com/openai/v1",
        openai_api_key=api_key,
        temperature=0.3,
        max_tokens=512
    )

    prompt = ChatPromptTemplate.from_template("""
    Use the following context to answer the question.
    If you don't know the answer, say "I don't know."

    Context:
    {context}

    Question: {question}
    """)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # --------------------------
    # ASK QUESTION
    # --------------------------
    st.divider()
    question = st.text_input("💬 Ask anything about your files:")

    if st.button("Get Answer") and question:
        with st.spinner("🤖 Thinking..."):
            answer = rag_chain.invoke(question)

        st.subheader("🧠 Answer")
        st.write(answer)

        # --------------------------
        # RECOMMENDATIONS
        # --------------------------
        st.subheader("✨ Related Sections")

        similar_docs = vectorstore.similarity_search(question, k=4)
        for doc in similar_docs:
            st.markdown(f"**📌 From:** `{doc.metadata.get('source_file')}`")
            st.write(doc.page_content[:400] + "...")
            st.divider()

        # --------------------------
        # FOLLOW-UP QUESTIONS
        # --------------------------
        st.subheader("💡 Suggested Follow-up Questions")

        q_prompt = f"Suggest 5 deeper follow-up questions for: '{question}'."
        follow = llm.invoke(q_prompt)
        st.write(follow.content)

else:
    st.info("⬆ Upload some files to get started.")
