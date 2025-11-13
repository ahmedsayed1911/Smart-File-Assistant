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

# -------------------------------------
# API KEY
# -------------------------------------
api_key = st.secrets.get("GROQ_API_KEY")
if not api_key:
    st.error("❌ Please add GROQ_API_KEY in Streamlit Secrets.")
    st.stop()

# -------------------------------------
# UI
# -------------------------------------
st.set_page_config(page_title="Smart File Assistant", layout="wide")
st.title("📘 Multi-File RAG System + Smart Recommendations")

uploaded_files = st.file_uploader(
    "Upload files: PDF / TXT / CSV",
    type=["pdf", "txt", "csv"],
    accept_multiple_files=True
)

# -------------------------------------
# FILE LOADING
# -------------------------------------
def load_file(uploaded_file):
    suffix = uploaded_file.name.split(".")[-1].lower()

    # Save temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name

    # Process file types
    if suffix == "pdf":
        loader = PyPDFLoader(path)
        docs = loader.load()

    elif suffix == "txt":
        loader = TextLoader(path)
        docs = loader.load()

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

    # Ensure all are LangChain Documents
    cleaned_docs = []
    for d in docs:
        if isinstance(d, Document):
            d.metadata["source_file"] = uploaded_file.name
            cleaned_docs.append(d)
        else:  # if raw dict (rare)
            cleaned_docs.append(
                Document(
                    page_content=d["page_content"],
                    metadata=d.get("metadata", {"source_file": uploaded_file.name})
                )
            )

    return cleaned_docs

# -------------------------------------
# BUILD RAG PIPELINE
# -------------------------------------
if uploaded_files:

    all_docs = []
    for f in uploaded_files:
        all_docs.extend(load_file(f))

    st.success(f"✔ Loaded {len(all_docs)} documents from {len(uploaded_files)} files.")

    # Split documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(all_docs)

    # Vector DB
    embedding = FastEmbedEmbeddings()
    vectorstore = Chroma.from_documents(chunks, embedding=embedding)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # LLM setup (Groq)
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

    # -------------------------------------
    # ASK (UI same as RAG System)
    # -------------------------------------
    st.divider()
    st.subheader("💬 Ask Your Question")

    question = st.text_area(
        "Type your question:",
        height=120,
        placeholder="Ask anything about your PDFs, CSVs, or TXT files..."
    )

    if st.button("🔍 Get Answer", use_container_width=True) and question:

        with st.spinner("🤖 Thinking..."):
            answer = rag_chain.invoke(question)

        # ------------------ ANSWER ------------------
        st.markdown("### 🧠 Answer")
        st.success(answer)

        # ------------------ RECOMMENDATIONS (Auto) ------------------
        st.markdown("### ✨ Related Sections (Recommendations)")
        related_docs = vectorstore.similarity_search(question, k=4)

        if len(related_docs) > 0:
            for doc in related_docs:
                st.info(
                    f"📌 **Source:** `{doc.metadata.get('source_file')}`\n\n"
                    f"{doc.page_content[:350]}..."
                )
        else:
            st.info("No related sections found.")

        # ------------------ FOLLOW-UP Questions ------------------
        st.markdown("### 💡 Suggested Follow-up Questions")

        follow_prompt = f"Suggest 5 deeper follow-up questions for: '{question}'."
        follow = llm.invoke(follow_prompt)

        st.warning(follow.content)

else:
    st.info("⬆ Upload files (PDF / TXT / CSV) to get started.")
