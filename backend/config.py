"""
Central configuration for StudyBuddy.

Model names live here and nowhere else. The previous version hardcoded
`llama-3.3-70b-versatile` in three separate files, so when Groq retired that
model the whole app broke and there was no single place to fix it.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- Models -------------------------------------------------------------
# Chat model used by the router and by every tool.
#
# Deliberately a "-latest" alias rather than a pinned version: this project
# broke in the first place because a pinned model (llama-3.3-70b-versatile)
# was retired out from under it. The alias tracks Google's current flash-lite.
#
# flash-lite was chosen over the larger flash models because it routed all
# four tools correctly in testing at ~3s/call, while gemini-3.8-flash caps the
# free tier at 20 requests/day — one afternoon of studying would exhaust it.
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-flash-lite-latest")

# Embedding model. IMPORTANT: this must match whatever the existing Chroma
# collection was built with. `gemini-embedding-001` emits 3072-dim vectors and
# the collection on disk is 3072-dim — changing this model means you must
# delete backend/chroma_db/ and re-index every document.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")

# --- Storage ------------------------------------------------------------
CHROMA_DIR = os.getenv("CHROMA_DIR", "backend/chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "studybuddy_docs")
DB_PATH = os.getenv("DB_PATH", "backend/rag_app.db")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")


def require_api_key() -> None:
    """Fail loudly at startup instead of mid-request if the key is missing."""
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Add it to your .env file at the "
            "project root. Get a key at https://aistudio.google.com/apikey"
        )
