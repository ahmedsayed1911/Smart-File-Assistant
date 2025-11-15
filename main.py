import streamlit as st
import os
import tempfile
import pandas as pd
import hashlib
import json
from typing import List

# faster PDF reader
from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# -------------------------------
# Config
# -------------------------------
PERSIST_DIR = "chroma_db"
PROCESSED_FILES_META = os.path.join(PERSIST_DIR, "files.json")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
RETRIEVE_K = 2

st.set_page_config(page_title="Smart File Assistant — Fast", layout="wide")
st.title("📘 Smart File Assistant — Fast + Persistent RAG")

# -------------------------------------
# API KEY (Groq)
# -------------------------------------
api_key = st.secrets.get("GROQ_API_KEY")
if not api_key:
    st.error("❌ Please add GROQ_API_KEY to Streamlit Secrets.")
    st.stop()

# -------------------------------------
# Helpers
# -------------------------------------

def file_hash_bytes(content: bytes) -> str:
    h = hashlib.sha256()
    h.update(content)
    return h.hexdigest()


def compute_uploads_fingerprint(uploaded_files) -> dict:
    meta = {}
    for f in uploaded_files:
        b = f.read()
        h = file_hash_bytes(b)
        meta[f.name] = {"hash": h, "size": len(b)}
        # rewind for later use
        f.seek(0)
    return meta


def read_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    pages = []
    for p in reader.pages:
        text = p.extract_text() or ""
        pages.append(text)
    return "\n".join(pages)


def load_file_to_docs(uploaded_file) -> List[Document]:
    suffix = uploaded_file.name.split(".")[-1].lower()

    # save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{suffix}') as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name

    docs = []
    if suffix == 'pdf':
        text = read_pdf_text(path)
        docs = [Document(page_content=text, metadata={"source_file": uploaded_file.name})]

    elif suffix == 'txt':
        text = open(path, 'r', encoding='utf-8', errors='ignore').read()
        docs = [Document(page_content=text, metadata={"source_file": uploaded_file.name})]

    elif suffix == 'csv':
        df = pd.read_csv(path)
        text = df.to_string()
        docs = [Document(page_content=text, metadata={"source_file": uploaded_file.name})]

    else:
        return []

    return docs


# -------------------------------------
# UI: Upload
# -------------------------------------
uploaded_files = st.file_uploader(
    "Upload your files (PDF / TXT / CSV)",
    type=["pdf", "txt", "csv"],
    accept_multiple_files=True
)

# ensure persist dir exists
os.makedirs(PERSIST_DIR, exist_ok=True)

# -------------------------------------
# If files uploaded: build or reuse DB
# -------------------------------------
if uploaded_files:
    st.info("🔁 Checking files and preparing (fast path enabled)")

    fingerprint = compute_uploads_fingerprint(uploaded_files)

    # check if we've already processed same set
    processed = {}
    if os.path.exists(PROCESSED_FILES_META):
        try:
            with open(PROCESSED_FILES_META, 'r', encoding='utf-8') as f:
                processed = json.load(f)
        except Exception:
            processed = {}

    # if same fingerprint, reuse existing DB
    if processed == fingerprint and os.listdir(PERSIST_DIR):
        st.success("✔ Using existing vector DB (no reprocessing).")
        embedding = FastEmbedEmbeddings()
        vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding)
    else:
        # process files and build vector store (this runs once)
        st.info("📦 Processing files — this happens only once per changed upload set.")

        all_docs = []
        progress = st.progress(0)
        total = len(uploaded_files)
        i = 0
        for f in uploaded_files:
            docs = load_file_to_docs(f)
            all_docs.extend(docs)
            i += 1
            progress.progress(int(i / total * 100))

        # split docs (larger chunk size -> fewer embeddings)
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = splitter.split_documents(all_docs)

        st.info(f"🔢 Created {len(chunks)} chunks — computing embeddings & building Chroma DB...")

        embedding = FastEmbedEmbeddings()
        # build persistent chroma DB
        vectorstore = Chroma.from_documents(
            chunks,
            embedding_function=embedding,
            persist_directory=PERSIST_DIR,
            collection_name="my_docs"
        )
        # persist and save fingerprints
        try:
            vectorstore.persist()
            with open(PROCESSED_FILES_META, 'w', encoding='utf-8') as f:
                json.dump(fingerprint, f, ensure_ascii=False, indent=2)
        except Exception as e:
            st.warning(f"Could not persist vector DB: {e}")

    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVE_K})

    # -------------------------------------
    # LLM
    # -------------------------------------
    llm = ChatOpenAI(
        model="llama-3.3-70b-versatile",
        openai_api_base="https://api.groq.com/openai/v1",
        openai_api_key=api_key,
        temperature=0.3,
        max_tokens=512
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are a multilingual AI assistant with automatic translation.

Rules:
1. Detect the language of the user's question.
2. Translate context internally if needed.
3. Answer ONLY in the user's language.
4. Ensure accuracy. Do NOT hallucinate.
5. If the answer is not in the context, say the equivalent of "I don't know" in the user's language.

Context:
{context}

User question:
{question}

Your answer (in the user's language):
"""
    )

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # -------------------------------------
    # QUESTION UI
    # -------------------------------------
    st.divider()
    st.subheader("💬 Ask your question")

    question = st.text_area(
        "Type your question (Arabic / English / any language):",
        height=120,
        placeholder="Example: لخص البيانات — Summarize the dataset — Résume le fichier..."
    )

    if st.button("🔍 Get Answer", use_container_width=True) and question:
        with st.spinner("🤖 Thinking and translating..."):
            answer = rag_chain.invoke(question)

        st.markdown("### 🧠 Answer")
        st.success(answer)

        st.markdown("### ✨ Related Sections")
        related_docs = vectorstore.similarity_search(question, k=RETRIEVE_K)

        for doc in related_docs:
            st.info(
                f"📌 **Source:** `{doc.metadata.get('source_file')}`\n\n"
                f"{doc.page_content[:350]}..."
            )

        st.markdown("### 💡 Suggested Follow-up Questions")
        follow_prompt = f"Suggest 5 follow-up questions for: '{question}' — in the same language."
        follow = llm.invoke(follow_prompt)
        st.warning(getattr(follow, 'content', str(follow)))

else:
    st.info("⬆ Upload files to start.")
