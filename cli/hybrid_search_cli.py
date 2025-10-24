import argparse
from lib.hybrid_search import (
    normalize_scores
)

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparser.add_parser(name="normalize", help="Normalize the search score")
    normalize_parser.add_argument(
        "score", type=float, nargs="+", help="The score to be normalized"
    )


    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized_scores = normalize_scores(args.score)
            for score in normalized_scores:
                print(f"* {score:.4f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()