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
# API KEY (Groq)
# -------------------------------------
api_key = st.secrets.get("GROQ_API_KEY")
if not api_key:
    st.error("❌ Please add GROQ_API_KEY to Streamlit Secrets.")
    st.stop()

# -------------------------------------
# UI
# -------------------------------------
st.set_page_config(page_title="Smart File Assistant", layout="wide")
st.title("📘 Smart File Assistant — Multilingual RAG + Auto-Translation + Recommendations")

uploaded_files = st.file_uploader(
    "Upload your files (PDF / TXT / CSV)",
    type=["pdf", "txt", "csv"],
    accept_multiple_files=True
)


# -------------------------------------
# FILE LOADER
# -------------------------------------
def load_file(uploaded_file):
    suffix = uploaded_file.name.split(".")[-1].lower()

    # save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name

    # PDF
    if suffix == "pdf":
        loader = PyPDFLoader(path)
        docs = loader.load()

    # TXT
    elif suffix == "txt":
        docs = TextLoader(path).load()

    # CSV → convert to text
    elif suffix == "csv":
        df = pd.read_csv(path)
        text = df.to_string()
        docs = [Document(page_content=text, metadata={"source_file": uploaded_file.name})]

    else:
        return []

    # Convert all to Document
    cleaned = []
    for d in docs:
        if isinstance(d, Document):
            d.metadata["source_file"] = uploaded_file.name
            cleaned.append(d)
        else:
            cleaned.append(
                Document(
                    page_content=d.get("page_content", str(d)),
                    metadata=d.get("metadata", {"source_file": uploaded_file.name})
                )
            )
    return cleaned


# -------------------------------------
# PROCESS FILES
# -------------------------------------
if uploaded_files:

    all_docs = []
    for f in uploaded_files:
        all_docs.extend(load_file(f))

    st.success(f"✔ Loaded {len(all_docs)} documents from {len(uploaded_files)} files.")

    # split docs
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(all_docs)

    # embeddings + vector DB
    embedding = FastEmbedEmbeddings()
    vectorstore = Chroma.from_documents(chunks, embedding=embedding)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # LLM
    llm = ChatOpenAI(
        model="llama-3.3-70b-versatile",
        openai_api_base="https://api.groq.com/openai/v1",
        openai_api_key=api_key,
        temperature=0.3,
        max_tokens=512
    )

    # -------------------------------------
    # MULTILINGUAL + AUTO-TRANSLATION PROMPT
    # -------------------------------------
    prompt = ChatPromptTemplate.from_template("""
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
""")

    # RAG chain
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

        # ------------------ ANSWER ------------------
        st.markdown("### 🧠 Answer")
        st.success(answer)

        # ------------------ RECOMMENDATIONS ------------------
        st.markdown("### ✨ Related Sections")
        related_docs = vectorstore.similarity_search(question, k=4)

        for doc in related_docs:
            st.info(
                f"📌 **Source:** `{doc.metadata.get('source_file')}`\n\n"
                f"{doc.page_content[:350]}..."
            )

        # ------------------ FOLLOW-UP QUESTIONS ------------------
        st.markdown("### 💡 Suggested Follow-up Questions")

        follow_prompt = f"Suggest 5 follow-up questions for: '{question}' — in the same language."
        follow = llm.invoke(follow_prompt)

        st.warning(follow.content)

else:
    st.info("⬆ Upload files to start.")
