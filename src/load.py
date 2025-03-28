import json
import requests
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.base import BaseEmbedding

# Define a custom Deepseek embedding class
class DeepseekEmbedding(BaseEmbedding):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def get_text_embedding(self, text: str) -> list:
        # Replace the URL and payload with the correct endpoint and parameters per Deepseek's API docs.
        url = "https://api.deepseek.ai/embedding"  # Hypothetical endpoint
        payload = {"text": text, "api_key": self.api_key}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            # Assume the API returns a JSON object with an "embedding" field
            return response.json()["embedding"]
        else:
            raise Exception(f"Deepseek API error: {response.status_code} {response.text}")

    def get_texts_embedding(self, texts: list[str]) -> list:
        return [self.get_text_embedding(text) for text in texts]

# Load structured JSON data
with open(r"D:\\uni\Sem 6\\LLMs\\legal-compliance-search\\data\\processed_docs\\structured_compliance_data.json", "r", encoding="utf-8") as f:
    legal_docs = json.load(f)

# Convert structured JSON into a list of document texts
documents = []
for doc in legal_docs:
    for section in doc["sections"]:
        text = f"{section['title']}: {section['content']}"
        documents.append(text)

# Initialize the Deepseek embedding with your API key
embedding = DeepseekEmbedding(api_key="your_deepseek_api_key")

# Create a FAISS-backed vector store using the from_texts method with Deepseek embeddings
vector_store = FaissVectorStore.from_texts(texts=documents, embedding=embedding)

# Create a storage context and index, then persist the index
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
index.storage_context.persist(persist_dir="index_store")

print("Legal compliance documents indexed successfully.")
