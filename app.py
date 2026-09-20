import os
import re
from pathlib import Path

import chromadb
import streamlit as st

from dotenv import load_dotenv
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer

# UI functions
from ui import (
    load_css,
    show_header,
    show_sidebar,
    show_welcome,
    show_sources
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Employee Handbook RAG",
    page_icon="📄",
    layout="wide"
)


# ============================================================
# UI
# ============================================================

load_css()
show_sidebar()
show_header()


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "gemini-3.5-flash-lite"

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

COLLECTION_NAME = "employee_handbook"


# ------------------------------------------------------------
# RETRIEVAL SETTINGS
# ------------------------------------------------------------

TOP_K = 5

MAX_RETRIEVAL_DISTANCE = 1.25

MAX_SOURCE_DISTANCE = 1.10

SOURCE_DISTANCE_MARGIN = 0.25


# ------------------------------------------------------------
# PROJECT PATH
# ------------------------------------------------------------

PROJECT_FOLDER = Path(
    __file__
).resolve().parent

DATABASE_PATH = (
    PROJECT_FOLDER / "chroma_db"
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )


# ============================================================
# LOAD CHROMADB COLLECTION
# ============================================================

@st.cache_resource
def load_collection():

    client = chromadb.PersistentClient(
        path=str(DATABASE_PATH)
    )

    try:
        return client.get_collection(
            name=COLLECTION_NAME
        )

    except Exception:
        from src.index_document import build_index

        return build_index()


# ============================================================
# LOAD GEMINI CLIENT
# ============================================================

@st.cache_resource
def load_gemini_client():

    # Load .env for local development.
    load_dotenv(
        PROJECT_FOLDER / ".env"
    )

    # First try the local environment variable.
    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    # If it is not available, try Streamlit Secrets.
    if not api_key:

        try:
            api_key = st.secrets["GEMINI_API_KEY"]

        except Exception:
            api_key = None

    if not api_key:

        raise ValueError(
            "GEMINI_API_KEY was not found. "
            "Set it in your local .env file "
            "or Streamlit Cloud Secrets."
        )

    return genai.Client(
        api_key=api_key
    )


# ============================================================
# GET CONVERSATION HISTORY
# ============================================================

def get_conversation_history():

    if not st.session_state.messages:
        return ""

    recent_messages = (
        st.session_state.messages[-6:]
    )

    history = []

    for message in recent_messages:

        role = message["role"]

        content = message["content"]

        if role == "user":

            history.append(
                f"User: {content}"
            )

        elif role == "assistant":

            history.append(
                f"Assistant: {content}"
            )

    return "\n".join(history)


# ============================================================
# REWRITE FOLLOW-UP QUESTION
# ============================================================

def rewrite_question(
    question,
    gemini_client
):

    history = get_conversation_history()

    # No previous conversation
    if not history:
        return question

    prompt = f"""
You are helping an employee handbook
Retrieval-Augmented Generation system.

Rewrite the user's latest question into
a standalone search query.

Use the previous conversation to resolve
references such as:

- this
- that
- it
- they
- those
- the requirements
- what about
- how many
- does this apply

Rules:

1. Do NOT answer the question.
2. Do NOT add facts that are not in the
   conversation.
3. Preserve the user's intent.
4. Return ONLY the rewritten search query.
5. If the question is already standalone,
   return it unchanged.

Previous conversation:

{history}

Latest question:

{question}
"""

    response = (
        gemini_client
        .models
        .generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=100
            )
        )
    )

    rewritten = (
        response.text.strip()
        if response.text
        else question
    )

    return rewritten


# ============================================================
# RETRIEVE RELEVANT CHUNKS
# ============================================================

def retrieve_relevant_chunks(
    question,
    collection,
    embedding_model
):

    question_embedding = (
        embedding_model
        .encode(question)
        .tolist()
    )

    return collection.query(
        query_embeddings=[
            question_embedding
        ],
        n_results=TOP_K,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize_text(text):

    text = str(text).lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# GET QUESTION KEYWORDS
# ============================================================

def get_question_keywords(
    question
):

    stop_words = {

        "what",
        "is",
        "are",
        "the",
        "a",
        "an",
        "how",
        "many",
        "much",
        "do",
        "does",
        "can",
        "could",
        "would",
        "employees",
        "employee",
        "company",
        "their",
        "they",
        "them",
        "for",
        "of",
        "to",
        "and",
        "in",
        "on",
        "this",
        "that",
        "about",
        "provide",
        "provided",
        "receive",
        "receives",
        "get",
        "gets",
        "please",
        "tell",
        "me"
    }

    words = normalize_text(
        question
    ).split()

    return [
        word
        for word in words
        if (
            word not in stop_words
            and len(word) > 2
        )
    ]


# ============================================================
# KEYWORD RELEVANCE
# ============================================================

def keyword_relevance(
    question,
    document,
    metadata
):

    keywords = get_question_keywords(
        question
    )

    if not keywords:
        return 0.0

    document_text = normalize_text(
        document
    )

    section_text = normalize_text(
        metadata.get(
            "section",
            ""
        )
    )

    combined_text = (
        document_text
        + " "
        + section_text
    )

    matches = 0

    for keyword in keywords:

        if keyword in combined_text:
            matches += 1

    return (
        matches / len(keywords)
    )


# ============================================================
# FILTER RELEVANT SOURCES
# ============================================================

def filter_relevant_sources(
    results,
    question
):

    documents = (
        results["documents"][0]
    )

    metadatas = (
        results["metadatas"][0]
    )

    distances = (
        results["distances"][0]
    )

    candidates = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):

        relevance = keyword_relevance(
            question,
            document,
            metadata
        )

        candidates.append({
            "document": document,
            "metadata": metadata,
            "distance": distance,
            "keyword_relevance": relevance
        })

    if not candidates:

        return {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }

    # --------------------------------------------------------
    # Sort by vector distance
    # --------------------------------------------------------

    candidates.sort(
        key=lambda item: item["distance"]
    )

    best = candidates[0]

    best_distance = best["distance"]

    # --------------------------------------------------------
    # Reject if best result is too weak
    # --------------------------------------------------------

    if (
        best_distance
        > MAX_RETRIEVAL_DISTANCE
    ):

        return {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }

    # --------------------------------------------------------
    # Always keep the best result
    # --------------------------------------------------------

    selected = [
        best
    ]

    # --------------------------------------------------------
    # Add only genuinely related sources
    # --------------------------------------------------------

    for candidate in candidates[1:]:

        distance = candidate["distance"]

        relevance = (
            candidate["keyword_relevance"]
        )

        # Too far away
        if (
            distance
            > MAX_SOURCE_DISTANCE
        ):
            continue

        # Too far from the best result
        if (
            distance
            > best_distance
            + SOURCE_DISTANCE_MARGIN
        ):
            continue

        # No lexical overlap and noticeably
        # worse than the best result
        if (
            relevance == 0
            and distance
            > best_distance + 0.10
        ):
            continue

        selected.append(
            candidate
        )

    # Keep at most 3 sources
    selected = selected[:3]

    return {
        "documents": [[
            item["document"]
            for item in selected
        ]],

        "metadatas": [[
            item["metadata"]
            for item in selected
        ]],

        "distances": [[
            item["distance"]
            for item in selected
        ]]
    }


# ============================================================
# CREATE HANDBOOK CONTEXT
# ============================================================

def create_context(
    results
):

    context_parts = []

    for index, document in enumerate(
        results["documents"][0],
        start=1
    ):

        metadata = (
            results["metadatas"][0][
                index - 1
            ]
        )

        source = metadata.get(
            "source",
            "Unknown"
        )

        section = metadata.get(
            "section",
            "Unknown"
        )

        context_parts.append(
            f"""
SOURCE {index}

Document:
{source}

Section:
{section}

Content:
{document}
""".strip()
        )

    return "\n\n".join(
        context_parts
    )


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    question,
    results,
    gemini_client
):

    context = create_context(
        results
    )

    history = get_conversation_history()

    prompt = f"""
You are a helpful assistant answering
questions about an employee handbook.

Answer ONLY using the retrieved
handbook context.

Rules:

1. Do not use outside knowledge.

2. Do not invent facts.

3. Use previous conversation only to
   understand what the user's question
   refers to.

4. The retrieved handbook context is
   the authoritative source.

5. If the answer is not contained in
   the retrieved context, say exactly:

"I could not find this information in
the provided handbook."

6. Keep the answer concise and clear.

7. Do not mention embeddings, ChromaDB,
   retrieval, prompts, or internal systems.

8. When answering questions involving
   lists, requirements, rules, or procedures,
   preserve the order in which the information
   appears in the retrieved handbook context.

Previous conversation:

{history}

Retrieved handbook context:

{context}

Current question:

{question}
"""

    response = (
        gemini_client
        .models
        .generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=300
            )
        )
    )

    if not response.text:

        return (
            "I could not find this information "
            "in the provided handbook."
        )

    return response.text.strip()


# ============================================================
# DISPLAY PREVIOUS CONVERSATION
# ============================================================

for message in st.session_state.messages:

    if message["role"] == "user":

        with st.chat_message("user"):

            st.write(
                message["content"]
            )

    elif message["role"] == "assistant":

        with st.chat_message("assistant"):

            st.write(
                message["content"]
            )

            if message.get("sources"):

                show_sources(
                    message["sources"]
                )

            if message.get(
                "no_relevant_sources",
                False
            ):

                st.info(
                    "No relevant handbook "
                    "sections were found."
                )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about the employee handbook..."
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    question = question.strip()

    if not question:

        st.warning(
            "Please enter a question."
        )

    else:

        # ----------------------------------------------------
        # Save user message
        # ----------------------------------------------------

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        # ----------------------------------------------------
        # Display user message
        # ----------------------------------------------------

        with st.chat_message("user"):

            st.write(
                question
            )

        # ----------------------------------------------------
        # Assistant message
        # ----------------------------------------------------

        with st.chat_message("assistant"):

            answer = None

            filtered_results = None

            request_failed = False

            no_relevant_sources = False

            # ------------------------------------------------
            # CONTROLLED STATUS
            # ------------------------------------------------

            with st.status(
                "Starting...",
                expanded=False
            ) as status:

                try:

                    # ----------------------------------------
                    # Load components
                    # ----------------------------------------

                    status.update(
                        label="Loading handbook...",
                        state="running"
                    )

                    embedding_model = (
                        load_embedding_model()
                    )

                    collection = (
                        load_collection()
                    )

                    gemini_client = (
                        load_gemini_client()
                    )

                    # ----------------------------------------
                    # Rewrite follow-up question
                    # ----------------------------------------

                    status.update(
                        label="Understanding your question...",
                        state="running"
                    )

                    search_question = (
                        rewrite_question(
                            question,
                            gemini_client
                        )
                    )

                    # ----------------------------------------
                    # Retrieval
                    # ----------------------------------------

                    status.update(
                        label="Searching the handbook...",
                        state="running"
                    )

                    results = (
                        retrieve_relevant_chunks(
                            search_question,
                            collection,
                            embedding_model
                        )
                    )

                    # ----------------------------------------
                    # Validate retrieval
                    # ----------------------------------------

                    distances = (
                        results.get(
                            "distances",
                            [[]]
                        )[0]
                    )

                    if not distances:

                        answer = (
                            "I could not find this information "
                            "in the provided handbook."
                        )

                        no_relevant_sources = True

                    else:

                        best_distance = min(
                            distances
                        )

                        # ------------------------------------
                        # Hard rejection
                        # ------------------------------------

                        if (
                            best_distance
                            > MAX_RETRIEVAL_DISTANCE
                        ):

                            answer = (
                                "I could not find this information "
                                "in the provided handbook."
                            )

                            no_relevant_sources = True

                        else:

                            # --------------------------------
                            # Filter sources
                            # --------------------------------

                            status.update(
                                label=(
                                    "Selecting relevant "
                                    "handbook sections..."
                                ),
                                state="running"
                            )

                            filtered_results = (
                                filter_relevant_sources(
                                    results,
                                    search_question
                                )
                            )

                            if not filtered_results[
                                "documents"
                            ][0]:

                                answer = (
                                    "I could not find this "
                                    "information in the "
                                    "provided handbook."
                                )

                                no_relevant_sources = True

                            else:

                                # ----------------------------
                                # Generate answer
                                # ----------------------------

                                status.update(
                                    label="Generating answer...",
                                    state="running"
                                )

                                answer = (
                                    generate_answer(
                                        question,
                                        filtered_results,
                                        gemini_client
                                    )
                                )

                    # ----------------------------------------
                    # Finish
                    # ----------------------------------------

                    status.update(
                        label="Complete",
                        state="complete"
                    )

                except Exception as error:

                    request_failed = True

                    status.update(
                        label="Request failed",
                        state="error"
                    )

                    st.error(
                        "The RAG application could not "
                        "complete the request."
                    )

                    # Show actual error for debugging
                    st.exception(
                        error
                    )

            # ------------------------------------------------
            # DISPLAY ANSWER
            # ------------------------------------------------

            if not request_failed:

                st.write(
                    answer
                )

                # --------------------------------------------
                # DISPLAY SOURCES
                # --------------------------------------------

                if (
                    filtered_results
                    and filtered_results[
                        "documents"
                    ][0]
                ):

                    show_sources(
                        filtered_results
                    )

                elif no_relevant_sources:

                    st.info(
                        "No relevant handbook "
                        "sections were found."
                    )

                # --------------------------------------------
                # SAVE ASSISTANT MESSAGE
                # --------------------------------------------

                st.session_state.messages.append({

                    "role": "assistant",

                    "content": answer,

                    "sources": (
                        filtered_results
                        if filtered_results
                        else None
                    ),

                    "no_relevant_sources":
                        no_relevant_sources
                })