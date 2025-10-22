from sentence_transformers import SentenceTransformer
import numpy as np
from .search_utils import MOVIE_EMBEDDINGS_PATH
from pathlib import Path
from .search_utils import load_movies

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text:str):
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")

        embedding_test = self.model.encode([text])
        return embedding_test[0]
    
    def build_embeddings(self, documents):
        self.documents = documents
        docs_text = []
        for doc in documents:
            self.document_map[doc['id']] = doc
            docs_text.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(docs_text, show_progress_bar=True)

        np.save(MOVIE_EMBEDDINGS_PATH, self.embeddings)

        return self.embeddings
    
    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for doc in documents:
            self.document_map[doc['id']] = doc
        
        if Path(MOVIE_EMBEDDINGS_PATH).exists():
            embedding = np.load(MOVIE_EMBEDDINGS_PATH)
            if len(embedding) == len(documents):
                self.embeddings = embedding
                return self.embeddings
        return self.build_embeddings(documents)

    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first")

        embedding_query = self.generate_embedding(query)
        score = []
        for idx, emb in enumerate(self.embeddings):
            sim = cosine_similarity(embedding_query, emb)
            score.append((sim, self.documents[idx]))

        sorted_scores = sorted(score, key=lambda x: x[0], reverse=True)[:limit]
        return [{"score": val[0], "title": val[1]['title'], "description": val[1]['description']} for val in sorted_scores]


def verify_model():
    semantic_search = SemanticSearch()
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")

def embed_text(text: str):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    semantic_search = SemanticSearch()
    documents = load_movies()
    semantic_search.load_or_create_embeddings(documents)

    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {semantic_search.embeddings.shape[0]} vectors in {semantic_search.embeddings.shape[1]} dimensions")

def embed_query_text(query: str):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def search(query:str, limit:int=5):
    semantic_search = SemanticSearch()
    documents = load_movies()
    semantic_search.load_or_create_embeddings(documents)

    results = semantic_search.search(query, limit)
    for idx, result in enumerate(results):
        print(f"{idx + 1}. {result['title']} (score: {result['score']:.4f})\n")
        print(f"{result['description']}\n\n")