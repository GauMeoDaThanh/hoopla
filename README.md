### Learn RAG repo

1. Preprocessing

   - remove punctuation
   - tokenization
   - remove stop words
   - stemming

2. TF-IDF

   - invert index
   - term frequency
   - inverse document frequency

3. Keyword search
   - term frequency saturation: prevents any single term from dominating search results just because it appears many many times.
   - document length normalization: ensuring longer documents don't get unfair advantages over shorter, more focused ones
   - okapi BM25 algorithm

4. Semantic search
   - Vector operation
   - Dimension
   - Embedding
   - Dot product similarity
   - Cosine similarity
   - Locality-Sensitive Hashing (LSH)

5. Chunking
   - Chunking
   - Overlap
   - Semantic chunking
   - Embedding chunk
   - Edge cases
   - ColBERT & Late chunking

### How to run project - Ubuntu/WSL

1. `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. `git clone https://github.com/GauMeoDaThanh/hoopla.git`
3. `cd hoopla`
4. `uv venv`
5. `source .venv/bin/activate`
6. `uv run cli/keyword_search_cli.py`
