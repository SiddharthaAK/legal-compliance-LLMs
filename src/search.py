import numpy as np
from langchain_community.vectorstores import FAISS
from load_mistral import MistralEmbeddings
import faiss
import json
from pathlib import Path

INDEX_PATH = Path("D:/uni/Sem 6/LLMs/legal-compliance-search/embeddings/faiss_index_store").resolve()

# Load processed legal documents
with open("D:/uni/Sem 6/LLMs/legal-compliance-search/data/processed_docs/structured_compliance_data.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

# Initialize the embedding function
embedding_function = MistralEmbeddings()

# Load FAISS index
vector_store = FAISS.load_local(str(INDEX_PATH), embedding_function, allow_dangerous_deserialization=True)
faiss_index = vector_store.index

def search_faiss(query, top_k=5):
    """Searches FAISS index for the most relevant legal clauses based on the query."""
    query_embedding = embedding_function.embed_query(query)
    query_vector = np.array([query_embedding], dtype=np.float32)

    D, I = faiss_index.search(query_vector, k=top_k)
    
    results = [documents[i] for i in I[0] if i < len(documents)]  # Ensure valid indices
    return results

if __name__ == "__main__":
    while True:
        query = input("\n🔍 Enter your legal query (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        
        results = search_faiss(query)

        print("\n🔍 Top Matching Documents:")
        for idx, res in enumerate(results, start=1):
            print(f"\n🔹 Match {idx}: {res}")
