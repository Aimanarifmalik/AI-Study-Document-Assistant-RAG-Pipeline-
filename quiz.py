"""
"""

import json
from dotenv import load_dotenv
from llm_utils import get_llm

load_dotenv()  


def generate_summary(vectorstore) -> str:
    """
    Makes a short summary of the whole document.

    Why we can't just say "summarize the PDF" directly: the LLM never saw
    the whole PDF, only chunks. So instead we:
      1. Grab a good sample of chunks from across the document
      2. Glue their text together
      3. Ask the LLM to summarize THAT combined text

    This is a simplified version — a more advanced version for a bigger PDF
    would summarize chunk-by-chunk first, then summarize the summaries
    (this is literally called "map-reduce summarization" in LangChain, a
    good phrase to know for interviews).
    """

    # similarity_search with a generic query just grabs chunks that are
    # broadly representative of the document's main content.
    docs = vectorstore.similarity_search("main ideas and key points", k=8)
    combined_text = "\n\n".join(doc.page_content for doc in docs)

    prompt = (
        "Summarize the following document content in 5-7 bullet points. "
        "Keep it simple and clear, like study notes for a student.\n\n"
        f"{combined_text}"
    )

    response = get_llm(temperature=0.3).invoke(prompt)
    return response.content


def generate_quiz(vectorstore, num_questions: int = 5) -> list[dict]:
    """
    Generates flashcard-style quiz questions from the document.

    Returns a LIST of dictionaries, each shaped like:
        {"question": "...", "answer": "..."}

    Why we ask the LLM to respond in JSON:
    Normally the LLM just writes free-flowing text. But our Streamlit UI
    needs to loop through each question and answer separately to build
    flashcard cards. So we ask the LLM to give us back STRUCTURED data
    (JSON) instead of a paragraph — this way our Python code can reliably
    read "question" and "answer" as separate fields, instead of us trying
    to guess where one question ends and the next begins from plain text.
    """

    docs = vectorstore.similarity_search("important facts and concepts", k=8)
    combined_text = "\n\n".join(doc.page_content for doc in docs)

    prompt = (
        f"Based on the content below, create exactly {num_questions} quiz "
        "questions to help a student study. Mix easy and medium difficulty. "
        "Respond with ONLY a JSON array, no extra text, no markdown "
        "formatting, in exactly this shape:\n"
        '[{"question": "...", "answer": "..."}, ...]\n\n'
        f"Content:\n{combined_text}"
    )

    response = get_llm(temperature=0.3).invoke(prompt)
    raw_text = response.content.strip()

    # Sometimes the LLM wraps JSON in ```json ... ``` even when told not to.
    # This cleans that up before parsing, so our code doesn't crash.
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        raw_text = raw_text.replace("json", "", 1).strip()

    try:
        quiz_data = json.loads(raw_text)
    except json.JSONDecodeError:
        
        quiz_data = [{
            "question": "Could not generate quiz — try again.",
            "answer": "N/A",
        }]

    return quiz_data
