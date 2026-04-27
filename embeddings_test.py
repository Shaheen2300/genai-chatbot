import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Sentences
sentences = [
    "I love dogs",
    "Dogs are amazing animals",
    "The stock market crashed today"
]

# Generate embeddings
embeddings = []
for sentence in sentences:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=sentence
    )
    embeddings.append(response.data[0].embedding)

# Convert to numpy arrays
vec1 = np.array(embeddings[0])
vec2 = np.array(embeddings[1])
vec3 = np.array(embeddings[2])

# Cosine similarity function
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Compare similarities
print("Similarity (dogs vs dogs):", cosine_similarity(vec1, vec2))
print("Similarity (dogs vs stock):", cosine_similarity(vec1, vec3))