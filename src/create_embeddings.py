from sentence_transformers import SentenceTransformer

from load_document import DOCUMENT_PATH, load_document
from preprocess_document import preprocess_text
from section_chunk_document import split_by_sections


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


if __name__ == "__main__":
    # 1. Load the original document.
    original_text = load_document(DOCUMENT_PATH)

    # 2. Create a cleaned working copy in memory.
    processed_text = preprocess_text(original_text)

    # 3. Split the working copy into meaningful policy chunks.
    chunks = split_by_sections(processed_text)

    # 4. Load the embedding model.
    print("\nLoading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    # 5. Convert every chunk into an embedding vector.
    embeddings = model.encode(chunks)

    print("\nEmbeddings created successfully.\n")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Numbers in each embedding: {len(embeddings[0])}")

    print("\nFirst chunk:")
    print(chunks[0])

    print("\nFirst 10 numbers of its embedding:")
    print(embeddings[0][:10])