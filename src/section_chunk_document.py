import re

from .load_document import DOCUMENT_PATH, load_document


def split_by_sections(text):
    """
    Split the employee handbook into policy-based sections.

    Each numbered section becomes a separate chunk.
    """

    # Match numbered headings such as:
    # 1. INTRODUCTION
    # 2. ELIGIBILITY
    # 4. PAID TIME OFF
    section_pattern = r"(?m)(?=^\d+\.\s+[A-Z][A-Z ]*$)"

    chunks = re.split(section_pattern, text)

    # Remove empty chunks and surrounding whitespace.
    chunks = [
        chunk.strip()
        for chunk in chunks
        if chunk.strip()
    ]

    return chunks


if __name__ == "__main__":
    original_text = load_document(DOCUMENT_PATH)

    chunks = split_by_sections(original_text)

    print("\nSection-based chunking completed successfully.\n")

    print(f"Total chunks: {len(chunks)}")

    print("\n" + "-" * 70)

    for index, chunk in enumerate(chunks, start=1):
        print(f"\nCHUNK {index}")
        print("-" * 70)
        print(chunk[:500])