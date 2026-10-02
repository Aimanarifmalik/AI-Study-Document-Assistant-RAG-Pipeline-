A Retrieval-Augmented Generation application that turns static PDF materials into an interactive learning tool with source citations, document summaries, and auto-generated flashcards/quizzes.

Decoupled REST API Architecture: Features a FastAPI backend handling document vectorization, retrieval, and LLM orchestration independently from the Streamlit UI, validated via Pydantic schemas.

OCR-Enabled Document Ingestion: Hybrid reader combining native text extraction (PyMuPDF) with OCR fallback (RapidOCR) for scanned documents, parallelized across threads.

Zero-Cost, Hybrid Inference Stack: Combines high-speed inference via Groq (Llama 3.3) with locally run HuggingFace embeddings (all-MiniLM-L6-v2), eliminating vector retrieval API costs.

Source-Grounded Answering: Custom retrieval chain returns page numbers alongside answers to eliminate hallucination and ground outputs strictly in document context.

Structured Output Generation: JSON-constrained prompts generate interactive flashcard and quiz UI components directly from PDF contents.
