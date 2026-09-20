import re

from load_document import DOCUMENT_PATH, load_document


HEADING_PATTERN = re.compile(r"^\d+\.\s+[A-Z ]+$")


def split_by_sections(text):
    """
    Split text using numbered headings such as:
    4. PAID TIME OFF
    """

    chunks = []
    title_lines = []
    current_section = None

    for line in text.splitlines():
        clean_line = line.strip()

        # Check whether the current line is a numbered heading.
        is_heading = HEADING_PATTERN.match(clean_line)

        if is_heading:
            # Save the preceding policy section before starting a new one.
            if current_section is not None:
                chunks.append("\n".join(current_section).strip())

            # Add the document title to the first policy chunk.
            current_section = title_lines + [line]
            title_lines = []

        elif current_section is None:
            # These lines appear before the first numbered heading.
            title_lines.append(line)

        else:
            # Add normal policy text to the active section.
            current_section.append(line)

    # Save the final policy section.
    if current_section is not None:
        chunks.append("\n".join(current_section).strip())

    return chunks


if __name__ == "__main__":
    document_text = load_document(DOCUMENT_PATH)

    chunks = split_by_sections(document_text)

    print(f"\nTotal chunks created: {len(chunks)}\n")

    for index, chunk in enumerate(chunks, start=1):
        print("=" * 70)
        print(f"CHUNK {index}")
        print("=" * 70)
        print(chunk)
        print()