import streamlit as st


def load_css():
    """
    Load custom CSS styling for the application.
    """

    st.markdown(
        """
        <style>

        /* Main title */
        .app-title {
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.3rem;
        }

        /* Subtitle */
        .app-subtitle {
            color: #9ca3af;
            font-size: 1rem;
            margin-bottom: 1.5rem;
        }

        /* Welcome box */
        .welcome-box {
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid rgba(128, 128, 128, 0.25);
            margin-bottom: 1.5rem;
        }

        .welcome-title {
            font-size: 1.4rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }

        .welcome-text {
            color: #9ca3af;
            line-height: 1.6;
        }

        </style>
        """,
        unsafe_allow_html=True
    )


def show_header():
    """
    Display the main application header.
    """

    st.markdown(
        """
        <div class="app-title">
            📄 Employee Handbook RAG
        </div>

        <div class="app-subtitle">
            Ask questions about company policies, benefits,
            leave, remote work, and other handbook information.
        </div>
        """,
        unsafe_allow_html=True
    )


def show_sidebar():
    """
    Display the application sidebar.
    """

    with st.sidebar:

        st.markdown("## 📄 Employee Handbook")

        st.write(
            "Ask questions about company policies, benefits, "
            "leave, remote work, and other handbook information."
        )

        st.divider()

        st.markdown("### 💡 Example Questions")

        st.markdown(
            """
            • How many PTO days do employees receive?

            • How many sick-leave days are provided?

            • Can employees work remotely?

            • What are the remote-work requirements?
            """
        )

        st.divider()

        st.markdown("### 🛠️ Application")

        st.caption(
            "Powered by ChromaDB, SentenceTransformers "
            "and Gemini."
        )


def show_welcome():
    """
    Display the welcome message when there is no conversation.
    """

    if not st.session_state.messages:

        st.markdown(
            """
            <div class="welcome-box">

                <div class="welcome-title">
                    👋 Welcome to Employee Handbook RAG
                </div>

                <div class="welcome-text">
                    Ask questions about company policies,
                    benefits, paid time off, sick leave,
                    remote work, and other information
                    contained in the employee handbook.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


def show_sources(results):
    """
    Display retrieved handbook sources.

    The function accepts the ChromaDB result structure:

    {
        "documents": [[...]],
        "metadatas": [[...]],
        "distances": [[...]]
    }
    """

    if not results:
        return

    documents = results.get(
        "documents",
        [[]]
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]]
    )[0]

    distances = results.get(
        "distances",
        [[]]
    )[0]

    if not documents:
        return

    st.markdown("### 📚 Sources")

    for index, document in enumerate(
        documents,
        start=1
    ):

        metadata = metadatas[index - 1]

        distance = distances[index - 1]

        section = metadata.get(
            "section",
            "Unknown section"
        )

        source = metadata.get(
            "source",
            "Unknown source"
        )

        with st.expander(
            f"📖 {section} • Similarity distance: {distance:.2f}"
        ):

            st.caption(
                f"Source: {source}"
            )

            st.write(
                document
            )