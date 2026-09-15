"""
HTTP client for the FastAPI backend.
The frontend uses these functions instead of calling FastAPI directly.
"""

import os

import requests

# Where our backend lives
API_BASE = os.getenv("API_BASE", "http://localhost:8000")


# ============================================================
# /chat
# ============================================================
def send_message(question: str, session_id: str = None) -> dict:
    """POST a message to the agent, return the parsed response."""
    payload = {"question": question, "session_id": session_id}
    response = requests.post(f"{API_BASE}/chat", json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


# ============================================================
# /upload-doc
# ============================================================
def upload_doc(file) -> dict:
    """Upload a file (PDF/DOCX) to be indexed."""
    files = {"file": (file.name, file.getvalue(), file.type)}
    response = requests.post(f"{API_BASE}/upload-doc", files=files, timeout=300)
    response.raise_for_status()
    return response.json()


# ============================================================
# /list-docs
# ============================================================
def list_docs() -> list:
    """Get all indexed documents."""
    response = requests.get(f"{API_BASE}/list-docs", timeout=30)
    response.raise_for_status()
    return response.json()


# ============================================================
# /delete-doc
# ============================================================
def delete_doc(file_id: int) -> dict:
    """Delete a document by file_id."""
    payload = {"file_id": file_id}
    response = requests.post(f"{API_BASE}/delete-doc", json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


# ============================================================
# /chat-history
# ============================================================
def get_chat_history(session_id: str) -> dict:
    """Fetch saved turns for a session — used on page refresh."""
    response = requests.get(f"{API_BASE}/chat-history/{session_id}", timeout=30)
    response.raise_for_status()
    return response.json()


# ============================================================
# /clear-session
# ============================================================
def clear_session(session_id: str) -> dict:
    """Wipe a session's chat history."""
    response = requests.delete(f"{API_BASE}/clear-session/{session_id}", timeout=30)
    response.raise_for_status()
    return response.json()


# ============================================================
# /submit-quiz
# ============================================================
def submit_quiz(quiz_id: int, answers: dict) -> dict:
    """Send the user's chosen letters for grading. Answers: {index: 'A'}."""
    payload = {"quiz_id": quiz_id, "answers": {str(k): v for k, v in answers.items()}}
    response = requests.post(f"{API_BASE}/submit-quiz", json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


# ============================================================
# /quiz-score
# ============================================================
def get_quiz_score(session_id: str) -> dict:
    """Running quiz score for a session."""
    response = requests.get(f"{API_BASE}/quiz-score/{session_id}", timeout=30)
    response.raise_for_status()
    return response.json()