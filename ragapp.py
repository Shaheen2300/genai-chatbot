import streamlit as st
import tempfile
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

load_dotenv()

st.set_page_config(page_title="Smart RAG Chatbot", layout="wide")
st.title("🤖 Smart Chat with PDFs")

# -------------------------
# Session State
# -------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------
# Upload PDFs
# -------------------------
uploaded_files = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    all_docs = []

    with st.spinner("📄 Processing PDFs..."):
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                temp_path = tmp.name

            loader = PyPDFLoader(temp_path)
            docs = loader.load()

            for d in docs:
                d.metadata["source"] = uploaded_file.name

            all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )

    split_docs = splitter.split_documents(all_docs)

    embeddings = OpenAIEmbeddings()
    st.session_state.vectorstore = FAISS.from_documents(split_docs, embeddings)

    st.session_state.messages = []
    st.success("✅ PDFs processed!")

# -------------------------
# Chat Section
# -------------------------
if st.session_state.vectorstore:

    retriever = st.session_state.vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 20}
    )

    llm = ChatOpenAI(model="gpt-4o-mini")

    user_query = st.chat_input("Ask anything about your documents...")

    if user_query:

        # Save user message
        st.session_state.messages.append({"role": "user", "content": user_query})

        # Build conversation (last few messages only)
        conversation = ""
        for msg in st.session_state.messages[-6:]:
            conversation += f"{msg['role']}: {msg['content']}\n"

        with st.spinner("🤔 Thinking..."):

            # -------------------------
            # SUMMARY MODE
            # -------------------------
            if "summar" in user_query.lower():

                full_context = ""
                for doc in st.session_state.vectorstore.docstore._dict.values():
                    full_context += doc.page_content + "\n\n"

                response = llm.invoke(f"""
Summarize the document clearly in bullet points.

Focus on:
- main topic
- key findings
- important results
- conclusion

Document:
{full_context}
""")

                answer = response.content

            # -------------------------
            # SMART RAG MODE
            # -------------------------
            else:
                # 🔥 QUERY REWRITING (KEY FEATURE)
                rewrite_prompt = f"""
Rewrite the user's question into a standalone question.

Conversation:
{conversation}

Question:
{user_query}
"""

                rewritten_query = llm.invoke(rewrite_prompt).content

                # Debug (optional)
                st.caption(f"🔍 Rewritten query: {rewritten_query}")

                # Retrieval
                relevant_docs = retriever.invoke(rewritten_query)

                context = ""
                sources = set()

                for doc in relevant_docs:
                    src = doc.metadata.get("source", "Unknown")
                    sources.add(src)
                    context += f"\n\nSOURCE: {src}\n{doc.page_content}"

                # Final Answer
                response = llm.invoke(f"""
You are a smart assistant.

Use:
1. Conversation (for continuity)
2. Document context (primary source)

Be:
- clear
- conversational
- helpful

If answer is not fully in context, explain what is missing.

Conversation:
{conversation}

Context:
{context}

Question:
{user_query}
""")

                answer = response.content

                st.caption("📚 Sources: " + ", ".join(sorted(sources)))

        # Save assistant message
        st.session_state.messages.append({"role": "assistant", "content": answer})

# -------------------------
# Display Chat
# -------------------------
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])