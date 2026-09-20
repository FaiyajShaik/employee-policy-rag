import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer


MODEL_NAME = "gemini-3.5-flash-lite"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "employee_handbook"

PROJECT_FOLDER = Path(__file__).resolve().parent.parent
DATABASE_PATH = PROJECT_FOLDER / "chroma_db"


def retrieve_relevant_chunks(question, collection, embedding_model):
    """
    Convert the question into an embedding and retrieve
    the three most relevant chunks from ChromaDB.
    """

    question_embedding = embedding_model.encode(question).tolist()

    return collection.query(
        query_embeddings=[question_embedding],
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )


def create_context(results):
    """
    Combine retrieved chunks into context for Gemini.
    """

    context_parts = []

    for index, document in enumerate(results["documents"][0], start=1):
        metadata = results["metadatas"][0][index - 1]

        context_parts.append(
            f"""
SOURCE {index}
Document: {metadata["source"]}
Section: {metadata["section"]}

Content:
{document}
""".strip()
        )

    return "\n\n".join(context_parts)


if __name__ == "__main__":
    # Load the API key from the local .env file.
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY was not found in the .env file.")

    # Load the local embedding model.
    print("\nLoading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Connect to the stored ChromaDB knowledge base.
    client = chromadb.PersistentClient(path=str(DATABASE_PATH))
    collection = client.get_collection(name=COLLECTION_NAME)

    # Connect to Gemini.
    gemini_client = genai.Client(api_key=api_key)

    # Ask the user for a question.
    question = input("\nAsk a question about the employee handbook:\n> ").strip()

    if not question:
        print("Please enter a question.")
        raise SystemExit

    # Retrieve relevant chunks.
    results = retrieve_relevant_chunks(
        question,
        collection,
        embedding_model
    )

    context = create_context(results)

    # Create a grounded RAG prompt.
    prompt = f"""
You are a helpful assistant answering questions about an employee handbook.

Answer the question using only the retrieved context below.

Rules:
1. Do not use outside knowledge.
2. Do not invent facts.
3. If the answer is not available in the context, say:
   "I could not find this information in the provided handbook."
4. Give a concise, clear answer.

Retrieved context:
{context}

Question:
{question}
"""

    # Generate the final answer.
    response = gemini_client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=300
        )
    )

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(response.text)

    print("\n" + "=" * 70)
    print("RETRIEVED SOURCES")
    print("=" * 70)

    for index, metadata in enumerate(results["metadatas"][0], start=1):
        print(
            f"{index}. {metadata['source']} — "
            f"{metadata['section']}"
        )