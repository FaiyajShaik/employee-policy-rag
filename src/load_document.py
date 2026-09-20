from pathlib import Path


# Find the main project folder.
PROJECT_FOLDER = Path(__file__).resolve().parent.parent

# Build the exact path to our source document.
DOCUMENT_PATH = PROJECT_FOLDER / "data" / "rag_source_employee_handbook.txt"


def load_document(file_path):
    """
    Read and return all text from a TXT document.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


if __name__ == "__main__":
    document_text = load_document(DOCUMENT_PATH)

    print("\nDocument loaded successfully.\n")
    print("-" * 60)
    print(document_text)
    print("-" * 60)
    print(f"\nTotal characters in document: {len(document_text)}")