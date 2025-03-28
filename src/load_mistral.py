import json
import faiss
import time
import numpy as np
from mistralai import Mistral
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain.schema import Document

# Set your Mistral API key directly
MISTRAL_API_KEY = "XHjNTKFaU9dNaGqkmMhuBE6WXRGAb7S0"  # Replace with your real API key

INDEX_PATH = "D:\\uni\\Sem 6\\LLMs\\legal-compliance-search\\embeddings\\faiss_index_store"

class MistralEmbeddings(Embeddings):
    def __init__(self, api_key: str = MISTRAL_API_KEY, model: str = "mistral-embed"):
        self.client = Mistral(api_key=api_key)
        self.model = model
        self.max_tokens = 16384
        self.batch_size = 4

    def count_tokens(self, text):
        return len(text.split())

    def split_batch(self, texts):
        batches, batch = [], []
        token_count = 0
        for text in texts:
            text_tokens = self.count_tokens(text)
            if token_count + text_tokens > self.max_tokens:
                if batch:
                    batches.append(batch)
                batch, token_count = [], 0
            batch.append(text)
            token_count += text_tokens
        if batch:
            batches.append(batch)
        return batches

    def embed_documents(self, texts):
        embeddings = []
        for batch in self.split_batch(texts):
            success = False
            while not success:
                try:
                    response = self.client.embeddings.create(model=self.model, inputs=batch)
                    embeddings.extend([item.embedding for item in response.data])
                    success = True
                    time.sleep(2)
                except Exception as e:
                    error_message = str(e).lower()
                    if "rate limit exceeded" in error_message:
                        print("Rate limit hit. Waiting 10 seconds...")
                        time.sleep(10)
                    elif "too many tokens" in error_message:
                        print("Too many tokens. Reducing batch size...")
                        self.batch_size = max(2, self.batch_size // 2)
                        batch = batch[: self.batch_size]
                    else:
                        print(f"Skipping batch due to API error: {e}")
                        success = True  
        return embeddings

    def embed_query(self, text):
        response = self.client.embeddings.create(model=self.model, inputs=[text])
        return response.data[0].embedding

with open("D:\\uni\\Sem 6\\LLMs\\legal-compliance-search\\data\\processed_docs\\structured_compliance_data.json", "r", encoding="utf-8") as f:
    legal_docs = json.load(f)

documents = [f"{section['title']}: {section['content']}" for doc in legal_docs for section in doc["sections"]]

embedding_function = MistralEmbeddings()
embeddings = embedding_function.embed_documents(documents)

if not embeddings:
    print("❌ No embeddings were created. Check API responses.")
    exit(1)

text_embedding_pairs = list(zip(documents, embeddings))  # ✅ Fix: Pair each text with its embedding

vector_store = FAISS.from_embeddings(text_embedding_pairs, embedding_function)

# ✅ Store FAISS index in the new directory
vector_store.save_local(INDEX_PATH)

print(f"✅ Legal compliance documents indexed successfully and saved to {INDEX_PATH}")
