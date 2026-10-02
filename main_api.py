"""
main_api.py - Complete FastAPI Backend for RAG Assistant
"""

import os
import tempfile
import traceback
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from ingest import build_vectorstore_from_pdf
from rag_chain import build_qa_chain
from quiz import generate_summary, generate_quiz

app = FastAPI(
    title="AI Study Assistant API",
    description="REST API for Document Chat, Summarization, and Quiz Generation",
    version="1.0.0"
)

state = {
    "vectorstore": None,
    "qa_chain": None
}

class ChatRequest(BaseModel):
    question: str

class QuizRequest(BaseModel):
    num_questions: int = 5


@app.get("/")
def root():
    return {
        "status": "online",
        "message": "AI Study Assistant API is live",
        "pdf_loaded": state["vectorstore"] is not None
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Uploads a PDF, builds the FAISS vector store, and sets up the RAG chain."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        content = await file.read()
        tmp.write(content)
        tmp.close()

        result = build_vectorstore_from_pdf(tmp.name)
        
        if isinstance(result, tuple):
            vectorstore = result[0]
        else:
            vectorstore = result

        qa_chain = build_qa_chain(vectorstore)

        state["vectorstore"] = vectorstore
        state["qa_chain"] = qa_chain

    except Exception as e:
        error_msg = traceback.format_exc()
        print("\n--- UPLOAD ERROR TRACEBACK ---")
        print(error_msg)
        print("------------------------------\n")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        if 'tmp' in locals() and os.path.exists(tmp.name):
            os.remove(tmp.name)

    return {
        "status": "success",
        "filename": file.filename,
        "message": "PDF indexed successfully! You can now use /ask, /summarize, and /quiz endpoints."
    }


@app.post("/ask")
def ask_question(request: ChatRequest):
    """Query the uploaded PDF using RAG."""
    if not state["qa_chain"]:
        raise HTTPException(
            status_code=400, 
            detail="No PDF processed yet. Please upload a PDF via /upload first."
        )

    try:
        try:
            result = state["qa_chain"].invoke({"query": request.question})
        except Exception:
            result = state["qa_chain"].invoke({"input": request.question})

        if isinstance(result, dict):
            answer = result.get("answer") or result.get("result") or str(result)
            sources_docs = result.get("source_documents") or result.get("context") or []
            sources = [doc.metadata.get("page", 1) for doc in sources_docs if hasattr(doc, "metadata")]
        else:
            answer = str(result)
            sources = []

        return {
            "question": request.question,
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing QA chain: {str(e)}")


@app.get("/summarize")
def get_summary():
    """Generates a summary of the uploaded PDF."""
    if not state["vectorstore"]:
        raise HTTPException(status_code=400, detail="No PDF processed yet.")

    try:
        summary_text = generate_summary(state["vectorstore"])
        return {"summary": summary_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")


@app.post("/quiz")
def get_quiz(request: QuizRequest):
    """Generates quiz flashcards from the uploaded PDF."""
    if not state["vectorstore"]:
        raise HTTPException(status_code=400, detail="No PDF processed yet.")

    try:
        quiz_items = generate_quiz(state["vectorstore"], num_questions=request.num_questions)
        return {"quiz": quiz_items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating quiz: {str(e)}")