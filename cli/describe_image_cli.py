import argparse
from lib.describe_image import (
    describe_image_command
)

def main() -> None:
    parser = argparse.ArgumentParser(description="Described Image")
    parser.add_argument(
        "--image",
        type=str,
        help="The path to the image file",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="A text query to rewrite based on the image"
    )
    args = parser.parse_args()

    describe_image_command(args.image, args.query)


if __name__ == "__main__":
    main()
