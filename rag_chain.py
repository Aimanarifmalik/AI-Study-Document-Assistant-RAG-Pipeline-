"""
rag_chain.py

This file builds the "chat with your PDF" logic.

What happens every time the user asks a question:
  user question
      -> FAISS finds the most relevant chunks from the PDF (the "Retriever")
      -> those chunks + the question are put into a prompt template
      -> the LLM (Groq, running Llama) reads the prompt and writes an answer
      -> we return the answer AND the chunks it used (so the UI can show pages)

NOTE ON WHY WE DON'T USE "RetrievalQA":
Newer LangChain versions removed the old RetrievalQA helper from the main
package. Instead of depending on it, we wire the steps together ourselves
with plain, stable pieces (retriever + prompt + LLM). It's only ~10 lines,
it works on every LangChain version, and it's actually EASIER to explain in
an interview because you can see every step instead of a hidden helper.
"""

from llm_utils import get_llm
from langchain_core.prompts import PromptTemplate


class SimpleQAChain:
    """
    A tiny hand-made RAG chain. It has the same .invoke() shape the UI
    expects: give it {"query": "..."} and get back
    {"result": "...", "source_documents": [...]}.
    """

    def __init__(self, retriever, llm, prompt):
        self.retriever = retriever   # finds relevant chunks (FAISS search)
        self.llm = llm               # writes the answer (Groq / Llama)
        self.prompt = prompt         # the fill-in-the-blanks instruction

    def invoke(self, inputs: dict) -> dict:
        question = inputs["query"]

        # STEP 1: RETRIEVE — ask FAISS for the chunks closest in meaning
        # to the question.
        docs = self.retriever.invoke(question)

        # STEP 2: Glue the chunk texts together into one "context" string.
        context = "\n\n".join(doc.page_content for doc in docs)

        # STEP 3: Fill the template's {context} and {question} blanks.
        final_prompt = self.prompt.format(context=context, question=question)

        # STEP 4: GENERATE — send the filled prompt to the LLM.
        response = self.llm.invoke(final_prompt)

        return {"result": response.content, "source_documents": docs}


def build_qa_chain(vectorstore):
    """
    Takes the FAISS vectorstore (built in ingest.py) and returns a ready-to-
    use chain that can answer questions about the PDF.
    """

    # temperature=0 -> be factual and consistent, don't get creative.
    # get_llm() (in llm_utils.py) picks a Groq model that currently exists.
    llm = get_llm(temperature=0)

    # k=4 -> fetch the 4 most relevant chunks for every question.
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # We tell the LLM to use ONLY the context and admit when it doesn't
    # know. This reduces "hallucination" (making things up).
    prompt = PromptTemplate(
        template=(
            "You are a helpful study assistant. Use ONLY the context below "
            "to answer the question. If the answer isn't in the context, "
            "say 'That's not covered in this document.'\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer clearly and simply, like you're explaining it to a student."
        ),
        input_variables=["context", "question"],
    )

    return SimpleQAChain(retriever, llm, prompt)
