from pathlib import Path
import re

import chromadb
from sentence_transformers import SentenceTransformer

from .load_document import DOCUMENT_PATH, load_document
from .preprocess_document import preprocess_text
from .section_chunk_document import split_by_sections


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

COLLECTION_NAME = "employee_handbook"

# Find the main First_RAG_Project folder.
PROJECT_FOLDER = Path(__file__).resolve().parent.parent

# ChromaDB will create this folder automatically.
DATABASE_PATH = PROJECT_FOLDER / "chroma_db"


def get_section_name(chunk):
    """
    Extract the numbered heading from a chunk.

    Example:
    4. PAID TIME OFF
    """

    for line in chunk.splitlines():
        clean_line = line.strip()

        if re.match(r"^\d+\.\s+[A-Z ]+$", clean_line):
            return clean_line

    return "Document introduction"


def build_index():
    """
    Build or rebuild the employee handbook
    ChromaDB collection.

    Returns:
        ChromaDB collection
    """

    # 1. Load the original source document.
    original_text = load_document(DOCUMENT_PATH)

    # 2. Safely preprocess text in memory.
    processed_text = preprocess_text(original_text)

    # 3. Split text into policy-based chunks.
    chunks = split_by_sections(processed_text)

    # 4. Load the same embedding model used by the RAG app.
    print("\nLoading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    # 5. Create one embedding for every chunk.
    embeddings = model.encode(chunks).tolist()

    # 6. Connect to persistent ChromaDB.
    client = chromadb.PersistentClient(
        path=str(DATABASE_PATH)
    )

    # 7. Create the collection if it does not exist.
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    # 8. Store chunks, embeddings, IDs, and metadata.
    collection.upsert(
        ids=[
            f"chunk_{index}"
            for index in range(1, len(chunks) + 1)
        ],
        documents=chunks,
        embeddings=embeddings,
        metadatas=[
            {
                "source": DOCUMENT_PATH.name,
                "chunk_number": index,
                "section": get_section_name(chunk)
            }
            for index, chunk in enumerate(
                chunks,
                start=1
            )
        ]
    )

    print("\nDocument indexed successfully.")
    print(
        f"Chunks stored in ChromaDB: "
        f"{collection.count()}"
    )
    print(
        f"Database folder: {DATABASE_PATH}"
    )

    return collection


if __name__ == "__main__":
    build_index()