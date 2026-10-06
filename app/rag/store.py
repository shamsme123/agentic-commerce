import os
import voyageai
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()  # reads VOYAGE_API_KEY and PINECONE_API_KEY from .env

INDEX_NAME = "agentic-commerce-policies"
EMBED_MODEL = "voyage-3.5"  # default output size of this model is 1024 numbers
DIMENSION = 1024  # the index size MUST match the embedding size
METRIC = "cosine"  # how "closeness" of two vectors is measured

voyage = voyageai.Client()  # reads VOYAGE_API_KEY from the environment
pinecone_client = Pinecone(api_key=os.environ["PINECONE_API_KEY"])


def embed(texts: list[str], input_type: str) -> list[list[float]]:
    # turns each text into a list of 1024 numbers (an "embedding").
    # texts with similar MEANING get similar numbers, that is what makes search work.
    # input_type is "document" for text we store and "query" for the user's question,
    # voyage tunes the numbers slightly for each case.
    result = voyage.embed(texts, model=EMBED_MODEL, input_type=input_type)
    return result.embeddings


def get_index():
    # create the Pinecone index the first time, then just reuse it
    if not pinecone_client.has_index(INDEX_NAME):
        print(f"786 creating Pinecone index {INDEX_NAME}")
        pinecone_client.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric=METRIC,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pinecone_client.Index(INDEX_NAME)
