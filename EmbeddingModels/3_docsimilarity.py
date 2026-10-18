from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# Load Hugging Face embedding model
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]


query = "tell me about bumrah"


# Convert documents into vectors
doc_embeddings = embedding.embed_documents(documents)

# Convert query into vector
query_embedding = embedding.embed_query(query)


# Calculate cosine similarity
scores = cosine_similarity(
    [query_embedding],
    doc_embeddings
)[0]


# Find document with highest similarity
index, score = sorted(
    list(enumerate(scores)),
    key=lambda x: x[1]
)[-1]


print("Query:", query)
print("Most similar document:", documents[index])
print("Similarity score:", score)