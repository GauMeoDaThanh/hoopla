#!/usr/bin/env python3

import argparse
from lib.semantic_search import (
    embed_query_text,
    search,
    verify_model,
    embed_text,
    verify_embeddings,
)

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify the model loading")
    embed_test_parser = subparsers.add_parser("embed_text", help="Embed text")
    embed_test_parser.add_argument("text", type=str, help="Text to embed")

    subparsers.add_parser("verify_embeddings", help="Verify embeddings loading and shape")

    embed_query_parser = subparsers.add_parser("embedquery", help="Embed query text")
    embed_query_parser.add_argument("query", type=str, help="Query text to embed")

    search_parse = subparsers.add_parser("search", help="Search for similar documents")
    search_parse.add_argument("query", type=str, help="Query text to search")
    search_parse.add_argument("--limit", type=int, default=5, help="Number of results to return")

    args = parser.parse_args()


    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            search(args.query, args.limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()