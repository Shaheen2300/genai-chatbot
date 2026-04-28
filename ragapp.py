import os
import hashlib
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

load_dotenv()

st.set_page_config(page_title="Chat with PDF (RAG)", layout="wide")
st.title("📄 Chat with Your PDF (RAG)")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "loaded_file_hash" not in st.session_state:
    st.session_state.loaded_file_hash = None

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    file_bytes = uploaded_file.read()
    file_hash = hashlib.md5(file_bytes).hexdigest()

    if file_hash != st.session_state.loaded_file_hash:
        st.session_state.chat_history = []
        st.session_state.vectorstore = None
        st.session_state.loaded_file_hash = file_hash

        temp_path = f"temp_{file_hash}.pdf"
        with open(temp_path, "wb") as f:
            f.write(file_bytes)

        with st.spinner("📄 Processing PDF..."):
            loader = PyPDFLoader(temp_path)
            documents = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
            docs = splitter.split_documents(documents)

            embeddings = OpenAIEmbeddings()
            st.session_state.vectorstore = FAISS.from_documents(docs, embeddings)
            os.remove(temp_path)

        st.success(f"✅ '{uploaded_file.name}' loaded! ({len(docs)} chunks created)")
    else:
        st.info(f"📎 '{uploaded_file.name}' already loaded.")

if st.session_state.vectorstore:
    llm = ChatOpenAI(model="gpt-4o-mini")
    user_query = st.chat_input("Ask something about your document...")

    if user_query:
        with st.spinner("🤔 Thinking..."):

            docs_and_scores = st.session_state.vectorstore.similarity_search_with_score(user_query, k=3)

           

            relevant_docs = [doc for doc, score in docs_and_scores if score < 0.5]
            

            if not relevant_docs:
                answer = "I don't know based on the document."
                context = ""
            else:
                context = "\n\n".join([doc.page_content for doc in relevant_docs])

                # ── STRONGER prompt — explicitly walls off outside knowledge ──
                response = llm.invoke(f"""
You are a document-only assistant. You have NO knowledge of the world.
You can ONLY use the text provided in the Context section below.
You MUST NOT use any knowledge from your training.
If the answer cannot be found word-for-word or by direct inference from the Context, 
you MUST respond with exactly: "I don't know based on the document."

Context:
---
{context}
---

Question: {user_query}

Answer (ONLY from the Context above):""")
                answer = response.content

            st.session_state.chat_history.append(("You", user_query))
            st.session_state.chat_history.append(("Bot", answer))

            if context:
                with st.expander("🔍 Retrieved Context"):
                    st.write(context)

for role, msg in st.session_state.chat_history:
    if role == "You":
        st.chat_message("user").write(msg)
    else:
        st.chat_message("assistant").write(msg)

if not uploaded_file:
    st.info("👆 Upload a PDF to begin")