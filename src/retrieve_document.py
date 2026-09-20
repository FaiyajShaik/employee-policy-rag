from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "employee_handbook"

PROJECT_FOLDER = Path(__file__).resolve().parent.parent
DATABASE_PATH = PROJECT_FOLDER / "chroma_db"


# Change this question later to test the RAG with other questions.
QUESTION = "How many paid time off days do employees receive?"


if __name__ == "__main__":
    # 1. Load the same embedding model used during indexing.
    print("\nLoading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    # 2. Convert the user's question into an embedding vector.
    question_embedding = model.encode(QUESTION).tolist()

    # 3. Connect to the stored ChromaDB database.
    client = chromadb.PersistentClient(path=str(DATABASE_PATH))

    # 4. Open the employee handbook collection.
    collection = client.get_collection(name=COLLECTION_NAME)

    # 5. Retrieve the three most relevant chunks.
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )

    print("\nQuestion:")
    print(QUESTION)

    print("\nMost relevant chunks:\n")

    for index, document in enumerate(results["documents"][0], start=1):
        metadata = results["metadatas"][0][index - 1]
        distance = results["distances"][0][index - 1]

        print("=" * 70)
        print(f"RESULT {index}")
        print(f"Source: {metadata['source']}")
        print(f"Section: {metadata['section']}")
        print(f"Distance: {distance:.4f}")
        print("-" * 70)
        print(document)
        print()