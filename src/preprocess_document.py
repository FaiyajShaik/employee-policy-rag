import re

from .load_document import DOCUMENT_PATH, load_document


def preprocess_text(text):
    """
    Prepare text for chunking without changing its meaning.

    This function:
    - standardizes line endings
    - removes indentation after line breaks
    - replaces repeated spaces with one space
    - limits excessive blank lines

    It does not edit or guess words in the original document.
    """

    processed_text = text.replace("\r\n", "\n")

    processed_text = processed_text.replace("\r", "\n")

    # Remove spaces or tabs at the beginning of lines.
    processed_text = re.sub(r"\n[ \t]+", "\n", processed_text)

    # Convert two or more spaces/tabs into one space.
    processed_text = re.sub(r"[ \t]{2,}", " ", processed_text)

    # Keep at most one blank line between paragraphs.
    processed_text = re.sub(r"\n{3,}", "\n\n", processed_text)

    return processed_text.strip()


if __name__ == "__main__":
    original_text = load_document(DOCUMENT_PATH)

    processed_text = preprocess_text(original_text)

    print("\nPreprocessing completed successfully.\n")

    print(f"Original character count: {len(original_text)}")
    print(f"Processed character count: {len(processed_text)}")

    print("\nProcessed-text preview:\n")

    print("-" * 70)

    print(processed_text[:800])

    print("-" * 70)