"""
llm_utils.py

ONE place that creates the Groq LLM for the whole project.

WHY THIS FILE EXISTS (simple explanation):
Groq keeps retiring old models and adding new ones. If we typed a model name
directly into rag_chain.py and quiz.py, the app would break every time Groq
retires it (that is exactly the 404 "model_not_found" error you saw).

So instead, this file:
  1. Uses the model you put in .env as GROQ_MODEL, if you set one.
  2. Otherwise asks Groq "which models can my key use right now?" and
     picks the best one from a preferred list.
That way the app keeps working even when Groq changes its lineup.
"""

import os
from langchain_groq import ChatGroq

# Best-first list of good chat models. The first one Groq says is active wins.
PREFERRED_MODELS = [
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3-32b",
    "llama-3.1-8b-instant",
]

# Words that mean "not a normal text-chat model" (speech, safety, etc.)
SKIP_WORDS = ["whisper", "guard", "tts", "orpheus", "embed", "compound", "vision"]

_cached_model_name = None  # remember the choice so we only ask Groq once


def _pick_model_name() -> str:
    global _cached_model_name
    if _cached_model_name:
        return _cached_model_name

    # Option 1: you chose a model yourself in .env  ->  GROQ_MODEL=some-model
    env_model = os.getenv("GROQ_MODEL")
    if env_model:
        _cached_model_name = env_model
        return env_model

    # Option 2: ask Groq which models are active for this API key.
    try:
        from groq import Groq
        client = Groq()  # reads GROQ_API_KEY from the environment
        active = [m.id for m in client.models.list().data]

        for name in PREFERRED_MODELS:
            if name in active:
                _cached_model_name = name
                return name

        # None of our favourites exist -> take any normal text model.
        for name in active:
            if not any(word in name.lower() for word in SKIP_WORDS):
                _cached_model_name = name
                return name
    except Exception:
        pass  # if the lookup fails, fall through to the default below

    _cached_model_name = PREFERRED_MODELS[0]
    return _cached_model_name


def get_llm(temperature: float = 0.0) -> ChatGroq:
    """Build a ChatGroq object using a model that actually exists today."""
    return ChatGroq(model=_pick_model_name(), temperature=temperature)
