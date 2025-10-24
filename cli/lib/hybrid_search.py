import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import load_movies


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        bm25_search = self._bm25_search(query, limit * 500)
        semantic_search = self.semantic_search.search_chunks(query, limit * 500)

        bm25_search_normalized = normalize_scores([search["score"] for search in bm25_search])
        semantic_search_normalized = normalize_scores([search["score"] for search in semantic_search])

        document_informations = {}
        for idx, search in enumerate(bm25_search):
            document_informations[search["id"]] = {
                "title": search["title"],
                "bm25_score": bm25_search_normalized[idx],
                "document": search["document"],
            }
        for idx, search in enumerate(semantic_search):
            document_informations[search["id"]]["semantic_score"] = semantic_search_normalized[idx]

        for doc_id, info in document_informations.items():
            bm25_score = info.get("bm25_score", 0)
            semantic_score = info.get("semantic_score", 0)

            hybrid_score_value = hybrid_score(bm25_score, semantic_score, alpha)
            document_informations[doc_id]["hybrid_score"] = hybrid_score_value
        
        return sorted(document_informations.values(), key=lambda x: x["hybrid_score"], reverse=True)[:limit]

    def rrf_search(self, query, k, limit=10):
        bm25_search = self._bm25_search(query, limit * 500)
        semantic_search = self.semantic_search.search_chunks(query, limit * 500)

        document_informations = {}
        for rank, search in enumerate(bm25_search, start=1):
            document_informations[search["id"]] = {
                "title": search["title"],
                "document": search["document"],
                "bm25_rank": rank,
            }
        for rank, search in enumerate(semantic_search, start=1):
            if search["id"] in document_informations:
                document_informations[search["id"]]["semantic_rank"] = rank
            else:
                document_informations[search["id"]] = {
                    "title": search["title"],
                    "document": search["document"],
                    "semantic_rank": rank,
                }

        for doc_id, info in document_informations.items():
            bm25_rank = info.get("bm25_rank", 0)
            semantic_rank = info.get("semantic_rank", 0)

            rrf_score_value = rrf_score(bm25_rank, k) + rrf_score(semantic_rank, k)
            document_informations[doc_id]["hybrid_score"] = rrf_score_value
        
        return sorted(document_informations.values(), key=lambda x: x["hybrid_score"], reverse=True)[:limit]

def normalize_scores(scores: list[float]) -> list[float]:
    max_score = max(scores)
    min_score = min(scores)

    if min_score == max_score:
        return [1.0 for _ in scores]

    return [(score - min_score) / (max_score - min_score) for score in scores]

def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score

def rrf_score(rank, k=60):
    return 1 / (k + rank)

def weighted_search_command(query, alpha=0.5, limit=5):
    movies = load_movies()
    hybrid_search = HybridSearch(movies)
    search_results = hybrid_search.weighted_search(query, alpha, limit)

    for idx, result in enumerate(search_results, start=1):
        print(f"{idx}. {result['title']}")
        print(f"   Hybrid Score: {result['hybrid_score']:.3f}")
        print(f"   BM25: {result.get('bm25_score', 0):.3f}   Semantic: {result.get('semantic_score', 0):.3f}")
        print(f"   Document: {result['document'][:100]}...")

def rrf_search_command(query, k=60, limit=5):
    movies = load_movies()
    hybrid_score = HybridSearch(movies)
    search_results = hybrid_score.rrf_search(query, k, limit)

    for idx, result in enumerate(search_results, start=1):
        print(f"{idx}. {result['title']}")
        print(f"   Hybrid Score: {result['hybrid_score']:.3f}")
        print(f"   BM25 Rank: {result.get('bm25_rank', 0)}   Semantic Rank: {result.get('semantic_rank', 0)}")
        print(f"   Document: {result['document'][:100]}...")
