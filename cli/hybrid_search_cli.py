import argparse
from lib.hybrid_search import (
    normalize_scores,
    weighted_search_command,
    rrf_search_command
)
from lib.gemini import generate_content

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparser.add_parser(name="normalize", help="Normalize the search score")
    normalize_parser.add_argument(
        "score", type=float, nargs="+", help="The score to be normalized"
    )

    weighted_search_parser = subparser.add_parser(
        name="weighted-search", help="Perform weighted hybrid search"
    )
    weighted_search_parser.add_argument(
        "query", type=str, help="The search query"
    )
    weighted_search_parser.add_argument(
        "--alpha", type=float, default=0.5, help="Weight for combining scores"
    )
    weighted_search_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return"
    )

    rrf_search_parser = subparser.add_parser(
        name="rrf-search", help="Perform RRF hybrid search"
    )
    rrf_search_parser.add_argument(
        "query", type=str, help="The search query"
    )
    rrf_search_parser.add_argument(
        "--k", type=int, default=60, help="RRF k parameter"
    )
    rrf_search_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return"
    )
    rrf_search_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Enhancement technique to apply"
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized_scores = normalize_scores(args.score)
            for score in normalized_scores:
                print(f"* {score:.4f}")
        case "weighted-search":
            weighted_search_command(args.query, args.alpha, args.limit)
        case "rrf-search":
            query = args.query
            if args.enhance == "spell":
                prompt = f"""Fix any spelling errors in this movie search query.

                Only correct obvious typos. Don't change correctly spelled words.

                Query: "{query}"

                If no errors, return the original query.
                Corrected:"""

                response = generate_content(prompt).text.strip()
                print(f"Enhanced query ({args.enhance}): {query} -> {response}\n")
                query = response
            if args.enhance == "rewrite":
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

                response = generate_content(prompt).text.strip()
                print(f"Enhanced query ({args.enhance}): {query} -> {response}\n")
                query = response
            if args.enhance == "expand":
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
                response = generate_content(prompt).text.strip()
                print(f"Enhanced query ({args.enhance}): {query} -> {response}\n")
                query = response

            rrf_search_command(query, args.k, args.limit, args.enhance)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()