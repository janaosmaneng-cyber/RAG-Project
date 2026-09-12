"""NeuroAtlas RAG Assistant — Streamlit frontend.

A chat-style UI: the user asks a question, optionally attaches an image
(Extended Track), and sees a grounded answer with its cited sources and,
if applicable, the image classification result.
"""

import re

import streamlit as st

from api_client import APIClientError, ask_question, check_health


def format_answer(text: str) -> str:
    """Convert inline '•' bullets from the model into proper markdown
    list items, while keeping any leading intro sentence as its own
    paragraph rather than turning it into a bullet."""
    parts = re.split(r"\s*•\s*", text)
    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) <= 1:
        return text  # no bullets found, leave as-is

    intro, *bullets = parts
    bullet_block = "\n\n".join(f"- {p}" for p in bullets)
    return f"{intro}\n\n{bullet_block}"


st.set_page_config(
    page_title="NeuroAtlas — RAG Assistant",
    page_icon="🧠",
    layout="centered",
)

# --- Custom styling: vibrant gradient / glittery look ------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Overall page — soft animated gradient wash */
    .stApp {
        background: linear-gradient(135deg, #f5f3ff 0%, #eef2ff 25%, #ecfeff 50%, #fdf4ff 75%, #fff1f2 100%);
        background-size: 400% 400%;
        animation: gradientShift 18s ease infinite;
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Caption under title */
    .stCaption {
        font-size: 0.97rem;
        color: #5b21b6;
        line-height: 1.55;
        font-weight: 500;
    }

    /* Chat bubbles — glassy cards with a glowing edge */
    [data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.75);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(168, 85, 247, 0.25);
        border-radius: 18px;
        padding: 16px 20px;
        margin-bottom: 14px;
        box-shadow: 0 4px 20px rgba(139, 92, 246, 0.12), 0 0 0 1px rgba(255,255,255,0.4) inset;
        transition: transform 0.15s ease;
    }
    [data-testid="stChatMessage"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 28px rgba(139, 92, 246, 0.2);
    }

    /* Assistant avatar circle — gradient glow */
    [data-testid="stChatMessageAvatarAssistant"] {
        background: linear-gradient(135deg, #7c3aed, #ec4899) !important;
        box-shadow: 0 0 12px rgba(236, 72, 153, 0.5);
    }

    /* User avatar circle — gradient glow */
    [data-testid="stChatMessageAvatarUser"] {
        background: linear-gradient(135deg, #06b6d4, #22d3ee) !important;
        box-shadow: 0 0 12px rgba(34, 211, 238, 0.5);
    }

    /* Sources expander */
    [data-testid="stExpander"] {
        border: 1px solid rgba(168, 85, 247, 0.3);
        border-radius: 14px;
        background: linear-gradient(135deg, rgba(243, 232, 255, 0.6), rgba(224, 242, 254, 0.6));
    }

    /* Info box (image classification) */
    [data-testid="stAlert"] {
        border-radius: 14px;
        border-left: 5px solid #ec4899;
        background: linear-gradient(90deg, rgba(236,72,153,0.08), rgba(139,92,246,0.08));
    }

    /* Chat input box */
    [data-testid="stChatInput"] {
        border-radius: 16px;
    }
    [data-testid="stChatInput"] textarea {
        border-radius: 16px;
        border: 2px solid transparent;
        background:
            linear-gradient(white, white) padding-box,
            linear-gradient(90deg, #a855f7, #ec4899, #06b6d4) border-box;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f3e8ff 0%, #e0f2fe 100%);
        border-right: 1px solid rgba(168, 85, 247, 0.2);
    }

    [data-testid="stSidebar"] h4 {
        font-family: 'Poppins', sans-serif;
    }

    /* Buttons — gradient pill */
    .stButton button {
        border-radius: 999px;
        border: none;
        background: linear-gradient(135deg, #7c3aed, #ec4899);
        color: white !important;
        font-weight: 600;
        box-shadow: 0 4px 14px rgba(236, 72, 153, 0.35);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .stButton button:hover {
        transform: translateY(-1px) scale(1.02);
        box-shadow: 0 6px 20px rgba(236, 72, 153, 0.45);
    }

    /* File uploader */
    [data-testid="stFileUploaderDropzone"] {
        border-radius: 16px;
        border: 2px dashed rgba(168, 85, 247, 0.4);
        background: rgba(255,255,255,0.5);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Header: sparkly gradient title + badge ---------------------------------
st.markdown(
    """
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:4px;">
        <span style="font-size:2.4rem; filter: drop-shadow(0 0 8px rgba(236,72,153,0.5));">🧠✨</span>
        <span style="font-family:'Poppins', sans-serif; font-size:2.4rem; font-weight:800;
                     background: linear-gradient(90deg, #7c3aed, #ec4899, #06b6d4);
                     -webkit-background-clip: text; background-clip: text; color: transparent;
                     letter-spacing: -0.5px;">
            NeuroAtlas
        </span>
        <span style="background: linear-gradient(135deg, #7c3aed, #ec4899); color:white; font-size:0.72rem;
                     padding:4px 12px; border-radius:20px; font-weight:700; letter-spacing:0.5px;
                     margin-left:4px; box-shadow: 0 2px 10px rgba(236,72,153,0.4);">
            ✨ RAG-POWERED
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption(
    "Ask a question about mental, neurodevelopmental, neurological, or "
    "sleep disorders. Answers are grounded in a curated document collection "
    "— optionally attach an image for Down syndrome, autism, or "
    "depression-related facial screening."
)
st.markdown(
    """
    <div style="height:3px; border-radius:3px; margin:14px 0 20px 0;
                background: linear-gradient(90deg, #7c3aed, #ec4899, #06b6d4, #7c3aed);
                background-size: 300% 100%; animation: gradientShift 6s linear infinite;"></div>
    """,
    unsafe_allow_html=True,
)

# --- Backend health check, shown once per session load -----------------
if "backend_healthy" not in st.session_state:
    st.session_state.backend_healthy = check_health()

if not st.session_state.backend_healthy:
    st.warning(
        "⚠️ Can't reach the backend right now. Make sure it's running "
        "(`uvicorn app.main:app --reload`) and refresh this page.",
        icon="⚠️",
    )

# --- Chat history --------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role", "content", "sources", "image_classification"}

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("image_classification"):
            ic = message["image_classification"]
            st.info(
                f"**Image classification** ({ic['dataset'].replace('_', ' ')} classifier): "
                f"predicted **{ic['predicted_class']}** "
                f"({ic['confidence']:.1%} confidence)"
            )

        if message.get("sources"):
            with st.expander(f"📚 Sources ({len(message['sources'])})"):
                for i, source in enumerate(message["sources"], start=1):
                    st.markdown(f"**[Source {i}]** {source}")

# --- Sidebar: about card + image attach + reset -----------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="background: rgba(255,255,255,0.75); backdrop-filter: blur(8px); padding:18px;
                    border-radius:16px; border:1px solid rgba(168,85,247,0.3); margin-bottom:16px;
                    box-shadow: 0 4px 16px rgba(139,92,246,0.15);">
            <h4 style="margin-top:0; font-family:'Poppins', sans-serif; font-weight:700;
                       background: linear-gradient(90deg, #7c3aed, #ec4899);
                       -webkit-background-clip: text; background-clip: text; color: transparent;">
                📖 About
            </h4>
            <p style="font-size:0.88rem; color:#475569; line-height:1.5; margin-bottom:0;">
                This assistant answers only from a curated document collection
                (WHO, NIMH, NINDS, NHLBI, NICHD, NIGMS, CDC, AASM materials).
                It is an informational tool, not a diagnostic one.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Moved here from the main content area so it never gets scrolled out
    # of view as the chat history grows — the sidebar stays put no matter
    # how long the conversation gets.
    st.markdown("**🖼️ Attach an image**")
    uploaded_image = st.file_uploader(
        "Optional: for Down syndrome / autism / depression screening",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
    )
    if uploaded_image is not None:
        st.image(uploaded_image, caption="Attached image", width=200)

    st.markdown("---")

    if st.button("🔄 Clear conversation"):
        st.session_state.messages = []
        st.rerun()

# --- Chat input ------------------------------------------------------------
question = st.chat_input("Ask a question...")

if question:
    # Show the user's message immediately.
    with st.chat_message("user"):
        st.markdown(question)
        if uploaded_image is not None:
            st.image(uploaded_image, width=150)

    st.session_state.messages.append({"role": "user", "content": question})

    # Call the backend, with a loading state while it runs.
    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating an answer..."):
            try:
                image_bytes = uploaded_image.getvalue() if uploaded_image is not None else None
                image_filename = uploaded_image.name if uploaded_image is not None else None

                result = ask_question(
                    question,
                    image_bytes=image_bytes,
                    image_filename=image_filename,
                )

                answer = format_answer(result.get("answer", ""))
                sources = result.get("sources", [])
                image_classification = result.get("image_classification")

                st.markdown(answer)

                if image_classification:
                    st.info(
                        f"**Image classification** "
                        f"({image_classification['dataset'].replace('_', ' ')} classifier): "
                        f"predicted **{image_classification['predicted_class']}** "
                        f"({image_classification['confidence']:.1%} confidence)"
                    )

                if sources:
                    with st.expander(f"📚 Sources ({len(sources)})"):
                        for i, source in enumerate(sources, start=1):
                            st.markdown(f"**[Source {i}]** {source}")

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "image_classification": image_classification,
                    }
                )

            except APIClientError as exc:
                error_message = f"⚠️ {exc}"
                st.error(error_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message}
                )