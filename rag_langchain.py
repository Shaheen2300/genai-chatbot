import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

# Load API key
load_dotenv()

# File path
PDF_PATH = r"C:\Users\sm397\Desktop\genai-project\sample.pdf"

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Step 1: Load or Create Vector DB
if os.path.exists("faiss_index"):
    print("✅ Loading existing vector DB...")
    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
else:
    print("⚡ Creating new vector DB...")

    # Load PDF
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    # Split into chunks
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(documents)

    # Create vector DB
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Save locally
    vectorstore.save_local("faiss_index")

# Step 2: Create retriever
retriever = vectorstore.as_retriever()

# Step 3: Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# Step 4: Ask questions in loop
print("\n🤖 RAG Chatbot Ready (type 'exit' to quit)\n")

while True:
    query = input("You: ")

    if query.lower() == "exit":
        break

    # Retrieve relevant docs
    relevant_docs = retriever.invoke(query)

    # Combine context
    context = "\n".join([doc.page_content for doc in relevant_docs])

    # Generate response
    response = llm.invoke(f"""
    Answer based only on the context below.

    Context:
    {context}

    Question:
    {query}
    """)

    print("\n🤖 Bot:", response.content)
    print("-" * 50)