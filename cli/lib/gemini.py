import os
import re
from dotenv import load_dotenv
from google import genai
import time
import json
from sentence_transformers import CrossEncoder

def generate_content(prompt):
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
        
    response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=prompt
    )
    return response

def generate_enhance_query(query, enhance_type):
    prompt = ""
    match enhance_type:
        case "spell":
            prompt = f"""Rewrite this movie search query to be more specific and searchable.

            Original: "{query}"

            Consider:
            - Common movie knowledge (famous actors, popular films)
            - Genre conventions (horror = scary, animation = cartoon)
            - Keep it concise (under 10 words)
            - It should be a google style search query that's very specific
            - Don't use boolean logic

            Examples:

            - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
            - "movie about bear in london with marmalade" -> "Paddington London marmalade"
            - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

            Rewritten query:"""
        case "expand":
            prompt = f"""Expand this movie search query with related terms.

            Add synonyms and related concepts that might appear in movie descriptions.
            Keep expansions relevant and focused.
            This will be appended to the original query.

            Examples:

            - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
            - "action movie with bear" -> "action thriller bear chase fight adventure"
            - "comedy with bear" -> "comedy funny bear humor lighthearted"

            Query: "{query}"
            """
        case "rewrite":
            prompt = f"""Rewrite this movie search query to be more specific and searchable.

            Original: "{query}"

            Consider:
            - Common movie knowledge (famous actors, popular films)
            - Genre conventions (horror = scary, animation = cartoon)
            - Keep it concise (under 10 words)
            - It should be a google style search query that's very specific
            - Don't use boolean logic

            Examples:

            - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
            - "movie about bear in london with marmalade" -> "Paddington London marmalade"
            - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

            Rewritten query:"""
        case _:
            return query
    response = generate_content(prompt).text.strip()
    return response

def rerank_response(query, documents, limit, rerank_method):
    match rerank_method:
        case "individual":
            for doc in documents:
                prompt = f"""Rate how well this movie matches the search query.

                Query: "{query}"
                Movie: {doc.get("title", "")} - {doc.get("document", "")}

                Consider:
                - Direct relevance to query
                - User intent (what they're looking for)
                - Content appropriateness

                Rate 0-10 (10 = perfect match).
                Give me ONLY the number in your response, no other text or explanation.

                Score:"""
                doc["rerank_score"] = float(generate_content(prompt).text.strip())
                time.sleep(3)
            documents = sorted(
                documents,
                key=lambda x: x.get("rerank_score", 0),
                reverse=True
            )[:limit]
        case "batch":
            documents_map = {idx: doc for idx, doc in enumerate(documents)}
            prompt = f"""Rank these movies by relevance to the search query.

            Query: "{query}"

            Movies:
            {documents_map}

            Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

            [75, 12, 34, 2, 1]
            """
            response = generate_content(prompt).text.strip()
            clean_response = re.sub(r"^```(?:json)?|```$", "", response, flags=re.IGNORECASE).strip()
            doc_rank_ids = json.loads(clean_response)

            for idx, doc_rank_id in enumerate(doc_rank_ids, start=1):
                documents[doc_rank_id]["rerank_rank"] = idx

            documents = sorted(
                documents, 
                key=lambda x: x.get("rerank_rank", float('inf'))
            )[:limit]
        case "cross_encoder":
            pairs = []
            for doc in documents:
                pairs.append([query, f"{doc.get('title', '')} - {doc.get('document', '')}"])

            cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
            scores = cross_encoder.predict(pairs)

            for idx, doc in enumerate(documents):
                doc["cross_encoder_score"] = scores[idx]
            
            documents = sorted(
                documents,
                key=lambda x: x.get("cross_encoder_score", 0),
                reverse=True
            )[:limit]

    return documents