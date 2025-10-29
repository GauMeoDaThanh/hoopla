import argparse
from urllib import response
from lib.hybrid_search import (
    normalize_scores,
    weighted_search_command,
    rrf_search_command
)
from lib.gemini import (
    generate_enhance_query,
    rerank_response
)

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
    rrf_search_parser.add_argument(
        "--rerank-method", type=str, choices=["individual", "batch", "cross_encoder"], help="Reranking method to use", default=None
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
            limit = args.limit * 5 if args.rerank_method else args.limit
            if args.enhance:
                query = generate_enhance_query(args.query, args.enhance)
                print(f"Enhanced query ({args.enhance}): {query} -> {response}\n")

            search_results = rrf_search_command(query, args.k,  limit)
            rerank_results = rerank_response(query, search_results, args.limit, args.rerank_method)

            for idx, result in enumerate(rerank_results, start=1):
                print(f"{idx}. {result['title']}")
                print(f"   Rerank Score: {result.get('rerank_score', 0):.3f}") if 'rerank_score' in result else ""
                print(f"   Rerank Rank : {result.get('rerank_rank', 0)}") if 'rerank_rank' in result else ""
                print(f"   Cross Encoder Score: {result.get('cross_encoder_score', 0):.3f}") if 'cross_encoder_score' in result else ""
                print(f"   Hybrid Score: {result['hybrid_score']:.3f}")
                print(f"   BM25 Rank: {result.get('bm25_rank', 0)}   Semantic Rank: {result.get('semantic_rank', 0)}")
                print(f"   Document: {result['document'][:100]}...")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()