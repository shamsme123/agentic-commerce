from pathlib import Path
from app.rag.store import embed, get_index

# knowledge/ sits in the repo root, two folders above this file
KNOWLEDGE_DIR = Path(__file__).resolve().parents[2] / "knowledge"


def load_chunks() -> list[dict]:
    # chunking = cutting a document into small pieces. small pieces keep each
    # search result focused. our policy files separate rules with a blank line,
    # so one paragraph = one chunk.
    chunks = []
    for file in sorted(KNOWLEDGE_DIR.glob("*.md")):
        paragraphs = file.read_text().split("\n\n")
        for number, paragraph in enumerate(paragraphs):
            text = paragraph.strip()
            if not text:
                continue
            # a fixed id (file name + position) means re-running this script
            # overwrites the same records instead of adding duplicates
            chunks.append({"id": f"{file.name}-{number}", "text": text, "source": file.name})
    return chunks


def ingest():
    chunks = load_chunks()
    print(f"786 loaded {len(chunks)} chunks from {KNOWLEDGE_DIR}")
    for chunk in chunks:
        print("   ", chunk["id"], "->", chunk["text"])

    # one embedding (list of 1024 numbers) per chunk
    vectors = embed([chunk["text"] for chunk in chunks], input_type="document")
    print(f"786 embedded, each vector has {len(vectors[0])} numbers")

    # store each vector with its text as metadata, so a search hit can give us
    # the original words back (the vector alone is only numbers)
    records = [
        {
            "id": chunk["id"],
            "values": vector,
            "metadata": {"text": chunk["text"], "source": chunk["source"]},
        }
        for chunk, vector in zip(chunks, vectors)
    ]
    index = get_index()
    index.upsert(vectors=records)
    print(f"786 upserted {len(records)} records into Pinecone")


if __name__ == "__main__":
    ingest()
