import argparse
from lib.hybrid_search import (
    normalize_scores,
    weighted_search_command,
    rrf_search_command
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

    rff_search_parser = subparser.add_parser(
        name="rrf-search", help="Perform RRF hybrid search"
    )
    rff_search_parser.add_argument(
        "query", type=str, help="The search query"
    )
    rff_search_parser.add_argument(
        "--k", type=int, default=60, help="RRF k parameter"
    )
    rff_search_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return"
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
            rrf_search_command(args.query, args.k, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()