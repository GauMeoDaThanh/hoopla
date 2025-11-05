import mimetypes
from .search_utils import (
    IMAGE_PATH
)
import os
from dotenv import load_dotenv
from google import genai
import types

load_dotenv()
api_key = os.getenv("gemini_api_key")
client = genai.Client(api_key=api_key)
model = "gemini-2.0-flash"

def describe_image_command(image_path, query):

    mime, _ = mimetypes.guess_type(image_path)
    mime = mime or "image/jpeg"

    image_data = None
    with open(IMAGE_PATH, "rb") as f:
        image_data = f.read()

    sys_prompt = "Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to: \
        - Synthesize visual and textual information \
        - Focus on movie-specific details (actors, scenes, style, etc.) \
        - Return only the rewritten query, without any additional commentary"
    
    parts = [
        sys_prompt,
        genai.types.Part.from_bytes(data=image_data, mime_type=mime),
        query.strip()
    ]

    response = client.models.generate_content(model=model, contents=parts)
    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")
    
