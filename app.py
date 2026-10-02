"""
app.py

The UI. Three things happen here:
  1. User uploads a PDF -> we build a vectorstore from it (ingest.py)
  2. User can CHAT with the PDF (rag_chain.py)
  3. User can get a SUMMARY or a QUIZ generated from it (quiz.py)

We use st.session_state to "remember" things between reruns. This matters
because Streamlit re-runs the ENTIRE script from top to bottom every single
time the user clicks ANYTHING. Without session_state, we'd lose the
vectorstore and have to re-process the PDF on every click — session_state
is Streamlit's way of saying "keep this variable alive across reruns."
"""

import tempfile
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

from ingest import build_vectorstore_from_pdf
from rag_chain import build_qa_chain
from quiz import generate_summary, generate_quiz


st.set_page_config(page_title="AI Study Assistant", page_icon="📚")
st.title("📚 AI Study/Document Assistant")
st.caption("Upload a PDF, chat with it, and auto-generate summaries + quizzes. (Free stack: Groq + local embeddings)")


if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (role, message) tuples

# ---- Step 1: Upload + process the PDF ----
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None and st.session_state.vectorstore is None:
    # A progress bar so you can SEE it working (OCR on image PDFs is slower).
    progress_bar = st.progress(0.0, text="Starting...")

    def update_progress(fraction, message):
        progress_bar.progress(min(fraction, 1.0), text=message)

    # Streamlit gives us the file in memory, but our PDF reader needs a
    # real file PATH on disk. tempfile creates a throwaway file so we
    # can save the upload there temporarily.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        vectorstore, stats = build_vectorstore_from_pdf(tmp_path, update_progress)
        progress_bar.progress(1.0, text="Building the search index...")
        st.session_state.vectorstore = vectorstore
        st.session_state.qa_chain = build_qa_chain(vectorstore)
        st.session_state.pdf_stats = stats
        progress_bar.empty()
        st.success("PDF processed! You can now chat, summarize, or generate a quiz.")
    except ValueError as e:
        progress_bar.empty()
        st.error(str(e))

# Show what was actually read from the PDF. If the numbers look tiny, the
# AI has nothing to search, which is the first thing to check when it says
# "not covered in this document".
if "pdf_stats" in st.session_state:
    s = st.session_state.pdf_stats
    with st.expander("What was read from your PDF (check this if answers look wrong)"):
        st.write(
            f"Pages: {s['total_pages']}  |  Pages with text: {s['pages_with_text']}  |  "
            f"Read with OCR: {s['pages_read_with_ocr']}  |  "
            f"Characters: {s['total_characters']:,}  |  Chunks: {s['total_chunks']}"
        )
        if not s["ocr_available"]:
            st.warning("OCR is not installed, so picture pages were skipped. Run: pip install rapidocr-onnxruntime")

# ---- Only show features once a PDF has been processed ----
if st.session_state.vectorstore is not None:

    tab_chat, tab_summary, tab_quiz = st.tabs(["💬 Chat", "📝 Summary", "🧠 Quiz"])

    # ----- CHAT TAB -----
    with tab_chat:
        # Show past messages first, so the conversation reads top to bottom.
        for role, message in st.session_state.chat_history:
            with st.chat_message(role):
                st.write(message)

        # st.chat_input gives a nice chat-style text box at the bottom.
        user_question = st.chat_input("Ask something about your document...")

        if user_question:
            st.session_state.chat_history.append(("user", user_question))
            with st.chat_message("user"):
                st.write(user_question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = st.session_state.qa_chain.invoke({"query": user_question})
                    answer = result["result"]
                    st.write(answer)

                    # Show which page(s) the answer came from — builds trust
                    # and is a nice touch to point out in an interview.
                    pages = {
                        doc.metadata.get("page", "?")
                        for doc in result["source_documents"]
                    }
                    st.caption(f"Source page(s): {sorted(pages)}")

            st.session_state.chat_history.append(("assistant", answer))

    # ----- SUMMARY TAB -----
    with tab_summary:
        if st.button("Generate Summary"):
            with st.spinner("Summarizing..."):
                summary = generate_summary(st.session_state.vectorstore)
            st.markdown(summary)

    # ----- QUIZ TAB -----
    with tab_quiz:
        num_q = st.slider("Number of questions", min_value=3, max_value=10, value=5)

        if st.button("Generate Quiz"):
            with st.spinner("Building your quiz..."):
                quiz_items = generate_quiz(st.session_state.vectorstore, num_q)
            st.session_state.quiz_items = quiz_items

        # Render each question as a flashcard using st.expander —
        # collapsed by default so the user has to "flip" it to see the answer,
        # just like a real flashcard.
        if "quiz_items" in st.session_state:
            for i, item in enumerate(st.session_state.quiz_items, start=1):
                with st.expander(f"Q{i}: {item['question']}"):
                    st.write(f"**Answer:** {item['answer']}")

else:
    st.info("Upload a PDF above to get started.")
