import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
# Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
print("RUNNING FILE...")
# Step 1: Your "documents" (like chunks from a PDF)
documents = [
    "Dogs are very loyal animals",
    "Cats are independent pets",
    "The stock market crashed yesterday",
    "Investing in stocks can be risky",
    "Dogs are commonly used as therapy animals"
]
# Step 2: Convert documents into embeddings
doc_embeddings = []
for doc in documents:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=doc
    )
    doc_embeddings.append(response.data[0].embedding)
# Step 3: User query
query = "Tell me about dogs"
query_embedding = client.embeddings.create(
    model="text-embedding-3-small",
    input=query
).data[0].embedding

# Step 4: Similarity function
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
# Step 5: Compare query with all documents
scores = []
for i, doc_emb in enumerate(doc_embeddings):
    score = cosine_similarity(query_embedding, doc_emb)
    scores.append((documents[i], score))

# Step 6: Sort by similarity
scores.sort(key=lambda x: x[1], reverse=True)

# Step 7: Print results
print("\n🔍 Most relevant results:\n")
for doc, score in scores:
    print(f"{doc} --> {score:.4f}")

# Step 8: Take top 2 documents
top_docs = [doc for doc, score in scores[:2]]

# Combine context
context = "\n".join(top_docs)

print("\n📄 Context being sent to LLM:\n")
print(context)

# Step 9: Send to LLM
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Answer based only on the provided context."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]
)

print("\n🤖 Final Answer:\n")
print(response.choices[0].message.content)
