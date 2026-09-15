"""
SQLite Persistence for StudyBuddy
----------------------------------
Handles all long-term storage:
- chat_logs: full conversation history per session
- documents: uploaded file metadata
- quiz_scores: quiz progress per session
"""

import json
import sqlite3
from typing import List, Dict, Optional

from langchain_core.messages import HumanMessage, AIMessage

from backend.config import DB_PATH


# ============================================================
# Connection helper
# ============================================================
def get_connection():
    """Open a connection with row-as-dict access."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row   # lets us access columns by name
    return conn


# ============================================================
# Schema setup — call once on app startup
# ============================================================
def create_tables():
    """Create all tables if they don't exist."""
    conn = get_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            user_query TEXT,
            agent_response TEXT,
            tool_used TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS quiz_scores (
            session_id TEXT PRIMARY KEY,
            correct INTEGER DEFAULT 0,
            total INTEGER DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Generated quizzes, including their answer keys. The answer key stays
    # server-side so the browser can't read the answers before grading.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS quizzes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            topic TEXT,
            questions_json TEXT NOT NULL,
            submitted INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()
    print("Tables created (or already exist).")


# ============================================================
# CHAT LOGS
# ============================================================
def insert_chat_log(session_id: str, user_query: str, agent_response: str, tool_used: str):
    """Save one user/agent turn."""
    conn = get_connection()
    conn.execute(
        "INSERT INTO chat_logs (session_id, user_query, agent_response, tool_used) "
        "VALUES (?, ?, ?, ?)",
        (session_id, user_query, agent_response, tool_used)
    )
    conn.commit()
    conn.close()


def get_chat_history(session_id: str) -> List:
    """
    Fetch full history for a session as a list of LangChain messages.
    Used by the graph to maintain conversation context.
    """
    conn = get_connection()
    rows = conn.execute(
        "SELECT user_query, agent_response FROM chat_logs "
        "WHERE session_id = ? ORDER BY created_at ASC",
        (session_id,)
    ).fetchall()
    conn.close()

    messages = []
    for row in rows:
        messages.append(HumanMessage(content=row["user_query"]))
        messages.append(AIMessage(content=row["agent_response"]))
    return messages


def get_chat_history_dicts(session_id: str) -> List[Dict]:
    """
    Fetch full history for a session as plain {role, content} dicts.
    Used by the frontend to rehydrate chat after a page refresh.
    """
    conn = get_connection()
    rows = conn.execute(
        "SELECT user_query, agent_response, tool_used FROM chat_logs "
        "WHERE session_id = ? ORDER BY created_at ASC",
        (session_id,)
    ).fetchall()
    conn.close()

    messages = []
    for row in rows:
        messages.append({"role": "user", "content": row["user_query"]})
        messages.append({
            "role": "assistant",
            "content": row["agent_response"],
            "tool_used": row["tool_used"],
        })
    return messages


def clear_session(session_id: str):
    """Delete all chat history, quizzes, and scores for a session."""
    conn = get_connection()
    conn.execute("DELETE FROM chat_logs WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM quiz_scores WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM quizzes WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()


# ============================================================
# QUIZZES
# ============================================================
def save_quiz(session_id: str, topic: str, questions: List[Dict]) -> int:
    """Store a generated quiz (with its answer key) and return its id."""
    conn = get_connection()
    cursor = conn.execute(
        "INSERT INTO quizzes (session_id, topic, questions_json) VALUES (?, ?, ?)",
        (session_id, topic, json.dumps(questions)),
    )
    quiz_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return quiz_id


def get_quiz(quiz_id: int) -> Optional[Dict]:
    """Fetch a stored quiz, answer key included."""
    conn = get_connection()
    row = conn.execute(
        "SELECT id, session_id, topic, questions_json, submitted FROM quizzes WHERE id = ?",
        (quiz_id,),
    ).fetchone()
    conn.close()

    if row is None:
        return None
    return {
        "id": row["id"],
        "session_id": row["session_id"],
        "topic": row["topic"],
        "questions": json.loads(row["questions_json"]),
        "submitted": bool(row["submitted"]),
    }


def mark_quiz_submitted(quiz_id: int):
    """Flag a quiz as graded so it can't be re-scored for extra points."""
    conn = get_connection()
    conn.execute("UPDATE quizzes SET submitted = 1 WHERE id = ?", (quiz_id,))
    conn.commit()
    conn.close()


# ============================================================
# DOCUMENTS
# ============================================================
def insert_document(filename: str) -> int:
    """Insert a new document record, return its auto-generated ID."""
    conn = get_connection()
    cursor = conn.execute(
        "INSERT INTO documents (filename) VALUES (?)", (filename,)
    )
    file_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return file_id


def list_documents() -> List[Dict]:
    """Return all uploaded documents."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, filename, uploaded_at FROM documents ORDER BY uploaded_at DESC"
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def delete_document_record(file_id: int):
    """Delete a document's metadata (call alongside delete_document in rag.py)."""
    conn = get_connection()
    conn.execute("DELETE FROM documents WHERE id = ?", (file_id,))
    conn.commit()
    conn.close()


# ============================================================
# QUIZ SCORES
# ============================================================
def get_quiz_score(session_id: str) -> Dict[str, int]:
    """Get the running quiz score for a session."""
    conn = get_connection()
    row = conn.execute(
        "SELECT correct, total FROM quiz_scores WHERE session_id = ?",
        (session_id,)
    ).fetchone()
    conn.close()

    if row is None:
        return {"correct": 0, "total": 0}
    return {"correct": row["correct"], "total": row["total"]}


def update_quiz_score(session_id: str, correct_increment: int, total_increment: int):
    """Increment the quiz score for a session (creates row if missing)."""
    current = get_quiz_score(session_id)
    new_correct = current["correct"] + correct_increment
    new_total = current["total"] + total_increment

    conn = get_connection()
    conn.execute("""
        INSERT INTO quiz_scores (session_id, correct, total, updated_at)
        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(session_id) DO UPDATE SET
            correct = excluded.correct,
            total = excluded.total,
            updated_at = CURRENT_TIMESTAMP
    """, (session_id, new_correct, new_total))
    conn.commit()
    conn.close()


# ============================================================
# Initialize on import
# ============================================================
create_tables()