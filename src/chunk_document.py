from load_document import DOCUMENT_PATH, load_document


def split_text(text, chunk_size=500, chunk_overlap=100):
    """
    Split text into chunks of a fixed number of characters.

    chunk_size:
        Maximum number of characters in one chunk.

    chunk_overlap:
        Number of repeated characters between neighboring chunks.
        Overlap prevents important context from being cut off.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end - chunk_overlap

    return chunks


if __name__ == "__main__":
    document_text = load_document(DOCUMENT_PATH)

    chunks = split_text(
        text=document_text,
        chunk_size=500,
        chunk_overlap=100
    )

    print(f"\nTotal chunks created: {len(chunks)}\n")

    for index, chunk in enumerate(chunks, start=1):
        print("=" * 70)
        print(f"CHUNK {index}")
        print("=" * 70)
        print(chunk)
        print()