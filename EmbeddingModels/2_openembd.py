from langchain_huggingface import HuggingFaceEmbeddings


embedding = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')


doc = "Delhi is the capital of India",
"PM MODI is prime minister of India",
"Muzaffarpur is a district in Biharpy"

vector = embedding.embed_documents(doc)

print(str(vector))