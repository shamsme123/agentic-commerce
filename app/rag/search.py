from app.rag.store import embed, get_index


def search_policy(query: str, top_k: int = 3) -> str:
    # 1. turn the question into numbers, the same way the policy chunks were
    query_vector = embed([query], input_type="query")[0]

    # 2. ask Pinecone for the top_k stored chunks whose numbers are closest
    index = get_index()
    result = index.query(vector=query_vector, top_k=top_k, include_metadata=True)

    # 3. give back the original policy words, not the numbers
    lines = []
    for match in result.matches:
        print(f"786 policy match score={match.score:.3f} source={match.metadata['source']}")
        lines.append(f"({match.metadata['source']}) {match.metadata['text']}")
    if not lines:
        return "No matching policy found."
    return "\n".join(lines)


if __name__ == "__main__":
    print(search_policy("Can I return my phone after opening it?"))
