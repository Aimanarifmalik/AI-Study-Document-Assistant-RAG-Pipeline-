"""
ingest.py

THE BIG IDEA (explained super simply):
An LLM like Llama can't read your whole PDF at once — it has a limited
"memory window" and doesn't know what's in YOUR specific document at all.
So instead of dumping the whole PDF into the chat, we do this:

  1. Read the PDF and cut it into small chunks (like cutting a book into
     paragraphs instead of handing someone the whole book).
  2. Turn each chunk into a list of numbers called an "embedding" — this is
     just a mathematical fingerprint of what that chunk MEANS (not the exact
     words, the meaning). Chunks about similar topics get similar fingerprints.
  3. Store all these fingerprints in a fast searchable index called FAISS.
  4. Later, when the user asks a question, we turn the QUESTION into a
     fingerprint too, and ask FAISS "which chunks have a fingerprint most
     similar to this question?" Those chunks are the most relevant parts
     of the PDF for answering that question.
  5. We hand ONLY those relevant chunks (not the whole PDF) to the LLM,
     along with the question, and say "answer using ONLY this information."

This whole process is called RAG = Retrieval-Augmented Generation.

IMPORTANT — WHY THIS VERSION IS 100% FREE:
Step 2 (turning text into embeddings) does NOT use OpenAI here. It uses a
model called "all-MiniLM-L6-v2" from HuggingFace, which downloads once
(a few hundred MB) and then runs directly on YOUR computer's CPU — no API
key, no internet needed after the first download, no cost, ever. Only the
LLM answering questions (step 5, handled in rag_chain.py / quiz.py) calls
an external API (Groq), and Groq's free tier covers that.
"""

from pdf_reader import extract_pages
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def build_vectorstore_from_pdf(pdf_path: str, progress_callback=None):
    """
    Takes a path to a PDF file on disk and returns (vectorstore, stats):
      vectorstore = searchable FAISS index built from that PDF's content
      stats       = numbers about what was read (pages, characters, OCR use)
    """

    # STEP 1: Read the PDF (see pdf_reader.py).
    # It reads normal text first, and uses OCR (reading words out of a
    # picture) for pages that are images, like lecture slides.
    # Each page becomes a "Document" with .page_content (text) and
    # .metadata (the page number, used to show sources later).
    pages, stats = extract_pages(pdf_path, progress_callback)

    # If we got no text at all, stop here with a clear message instead of
    # building an empty search index (that caused "not covered" answers).
    if stats["total_characters"] == 0:
        raise ValueError(
            "No text could be read from this PDF. It is probably scanned "
            "images and OCR is not installed. Run: pip install rapidocr-onnxruntime"
        )

    # STEP 2: Split the pages into smaller chunks.
    # Why not just use whole pages? Because a page might be too long, and
    # might contain MULTIPLE unrelated topics mixed together, which makes
    # the "fingerprint" (embedding) less precise. Smaller, focused chunks
    # give more accurate search results later.
    #
    # chunk_size=1000       -> roughly 1000 characters per chunk (~150-200 words)
    # chunk_overlap=150     -> each chunk repeats the last 150 characters of
    #                          the previous chunk, so we don't accidentally
    #                          cut a sentence/idea in half between two chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(pages)

    # STEP 3: Turn each chunk into an embedding using a FREE local model.
    # "sentence-transformers/all-MiniLM-L6-v2" is a small, fast, well-known
    # open-source embedding model. The first time you run this, it downloads
    # the model automatically (needs internet once). After that, it runs
    # fully offline on your CPU — no API key, no cost.
    # model_kwargs={"device": "cpu"} just forces it to run on CPU, which is
    # fine for a small project like this (no GPU required).
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    stats["total_chunks"] = len(chunks)
    return vectorstore, stats
