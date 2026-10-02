"""
pdf_reader.py

Reads a PDF and returns the text of every page.

THE PROBLEM THIS FIXES (simple explanation):
Some PDFs (like lecture slides) store each page as a PICTURE, not as real
text. A normal PDF reader looks for text, finds none, and gives back an
empty page. Then the search has nothing to search, and the AI says
"That's not covered in this document."

TWO-STEP SOLUTION:
  1. Try to read real text from the page (fast).
  2. If the page has almost no text, it is probably a picture. Turn the
     page into an image and use OCR (Optical Character Recognition, which
     means "a program that reads words out of a picture") to get the text.
"""

import pymupdf  # PyMuPDF: a fast PDF reader (better than pypdf on tricky files)
from langchain_core.documents import Document

MIN_TEXT_CHARS = 40  # fewer characters than this = treat the page as a picture

_ocr_engine = None  # created once, reused for every page (creating is slow)


def _get_ocr_engine():
    """Load the OCR engine one time. Returns None if it is not installed."""
    global _ocr_engine
    if _ocr_engine is not None:
        return _ocr_engine
    try:
        from rapidocr_onnxruntime import RapidOCR
        _ocr_engine = RapidOCR()
    except Exception:
        try:
            from rapidocr import RapidOCR  # newer package name
            _ocr_engine = RapidOCR()
        except Exception:
            _ocr_engine = False  # remember that OCR is unavailable
    return _ocr_engine


def _ocr_page(page) -> str:
    """Turn one PDF page into an image, then read the words from it."""
    engine = _get_ocr_engine()
    if not engine:
        return ""

    # zoom 2x = sharper picture = OCR reads small text better
    pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))
    image_bytes = pix.tobytes("png")

    result = engine(image_bytes)

    # Older package returns (list_of_[box, text, score], timing).
    # Newer package returns an object with a .txts attribute.
    if isinstance(result, tuple):
        lines = result[0] or []
        return "\n".join(item[1] for item in lines)
    if hasattr(result, "txts") and result.txts:
        return "\n".join(result.txts)
    return ""


def extract_pages(pdf_path: str, progress_callback=None):
    """
    Returns (documents, stats).
      documents = one LangChain Document per page that had any text
      stats     = numbers we show in the UI so you can see what happened
    """
    doc = pymupdf.open(pdf_path)
    total = len(doc)
    documents = []
    ocr_pages = 0

    for i in range(total):
        page = doc[i]
        text = page.get_text().strip()

        if len(text) < MIN_TEXT_CHARS:      # looks like a picture page
            ocr_text = _ocr_page(page).strip()
            if len(ocr_text) > len(text):
                text = ocr_text
                ocr_pages += 1

        if text:
            # page number starts at 1 (humans count from 1, Python from 0)
            documents.append(Document(page_content=text, metadata={"page": i + 1}))

        if progress_callback:
            progress_callback((i + 1) / total, f"Reading page {i + 1} of {total}")

    doc.close()

    stats = {
        "total_pages": total,
        "pages_with_text": len(documents),
        "pages_read_with_ocr": ocr_pages,
        "total_characters": sum(len(d.page_content) for d in documents),
        "ocr_available": bool(_get_ocr_engine()),
    }
    return documents, stats
