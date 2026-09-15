"""
FastAPI Backend for StudyBuddy Agent
-------------------------------------
Exposes the agent + RAG pipeline as HTTP endpoints.
Run with: uvicorn backend.main:app --reload
"""

import os
import uuid
import shutil
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from langchain_core.tracers.langchain import wait_for_all_tracers

from backend.config import UPLOAD_DIR
from backend.graph import run_agent
from backend.rag import index_docs, delete_document
from backend.db import (
    insert_chat_log,
    get_chat_history,
    get_chat_history_dicts,
    insert_document,
    list_documents,
    delete_document_record,
    clear_session,
    save_quiz,
    get_quiz,
    mark_quiz_submitted,
    get_quiz_score,
    update_quiz_score,
)
from backend.model import (
    ChatRequest,
    ChatResponse,
    QuizQuestionPublic,
    UploadResponse,
    DocumentInfo,
    DeleteRequest,
    DeleteResponse,
    QuizSubmission,
    GradedAnswer,
    QuizResult,
    ScoreResponse,
)

app = FastAPI(title="StudyBuddy Agent API", version="1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = (".pdf", ".docx")


@app.get("/")
def root():
    return {"message": "StudyBuddy Agent API is running 🚀"}


# --- /chat ---------------------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Send a user message; runs the LangGraph agent and returns its answer."""
    session_id = req.session_id or str(uuid.uuid4())

    # Only the last exchange, so graph state (and the LangSmith trace) stays small.
    history = get_chat_history(session_id)[-2:]

    try:
        result = run_agent(
            user_input=req.question,
            session_id=session_id,
            history=history,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")

    # Flush this run's LangSmith trace now, so it appears immediately rather
    # than on the next request. Costs a little latency per call.
    wait_for_all_tracers()

    answer = result.get("final_ans", "")
    tool_used = result.get("tool_call", "unknown")

    insert_chat_log(
        session_id=session_id,
        user_query=req.question,
        agent_response=answer,
        tool_used=tool_used,
    )

    # If a quiz was generated, persist it with its answer key and hand the
    # browser only the questions and options.
    quiz_id = None
    public_questions = None
    questions = result.get("quiz_questions") or []
    if questions:
        raw = result.get("raw_output") or {}
        topic = raw.get("topic", "") if isinstance(raw, dict) else ""
        quiz_id = save_quiz(session_id, topic, questions)
        public_questions = [
            QuizQuestionPublic(
                index=i,
                question=q.get("question", ""),
                options=q.get("options", []),
            )
            for i, q in enumerate(questions)
        ]

    return ChatResponse(
        answer=answer,
        session_id=session_id,
        tool_used=tool_used,
        quiz_id=quiz_id,
        quiz_questions=public_questions,
    )


# --- /submit-quiz --------------------------------------------------------
@app.post("/submit-quiz", response_model=QuizResult)
def submit_quiz(req: QuizSubmission):
    """Grade a quiz server-side and update the session's running score."""
    quiz = get_quiz(req.quiz_id)
    if quiz is None:
        raise HTTPException(status_code=404, detail="Quiz not found")

    questions = quiz["questions"]
    results: List[GradedAnswer] = []
    correct_count = 0

    for i, q in enumerate(questions):
        chosen = (req.answers.get(i) or "").strip().upper() or None
        correct = str(q.get("correct", "")).strip().upper()
        is_correct = chosen is not None and chosen == correct
        if is_correct:
            correct_count += 1
        results.append(
            GradedAnswer(
                index=i,
                question=q.get("question", ""),
                chosen=chosen,
                correct=correct,
                is_correct=is_correct,
                explanation=q.get("explanation", "") or "",
            )
        )

    # Only count toward the running score the first time a quiz is graded.
    if not quiz["submitted"]:
        update_quiz_score(quiz["session_id"], correct_count, len(questions))
        mark_quiz_submitted(req.quiz_id)

    return QuizResult(
        quiz_id=req.quiz_id,
        correct=correct_count,
        total=len(questions),
        results=results,
        session_score=get_quiz_score(quiz["session_id"]),
    )


# --- /quiz-score ---------------------------------------------------------
@app.get("/quiz-score/{session_id}", response_model=ScoreResponse)
def quiz_score(session_id: str):
    """Running quiz score for a session."""
    score = get_quiz_score(session_id)
    return ScoreResponse(session_id=session_id, **score)


# --- /upload-doc ---------------------------------------------------------
@app.post("/upload-doc", response_model=UploadResponse)
async def upload_doc(file: UploadFile = File(...)):
    """Upload a document, save it to disk, and index it into ChromaDB."""
    if not file.filename or not file.filename.lower().endswith(ALLOWED_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {ALLOWED_EXTENSIONS}",
        )

    # Keep the original name but make it unique, so re-uploading a file doesn't
    # overwrite the old one on disk while both rows live on in the database.
    base, ext = os.path.splitext(os.path.basename(file.filename))
    stored_name = f"{base}_{uuid.uuid4().hex[:8]}{ext}"
    file_path = os.path.join(UPLOAD_DIR, stored_name)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    file_id = insert_document(file.filename)

    try:
        index_docs(file_path, file_id)
    except Exception as e:
        # Roll back both the DB row and the file if indexing failed.
        delete_document_record(file_id)
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")

    return UploadResponse(
        file_id=file_id,
        filename=file.filename,
        message="Uploaded and indexed successfully ✅",
    )


# --- /list-docs ----------------------------------------------------------
@app.get("/list-docs", response_model=List[DocumentInfo])
def list_docs():
    """Return all uploaded documents."""
    return [
        DocumentInfo(
            id=d["id"],
            filename=d["filename"],
            uploaded_at=str(d["uploaded_at"]),
        )
        for d in list_documents()
    ]


# --- /delete-doc ---------------------------------------------------------
@app.post("/delete-doc", response_model=DeleteResponse)
def delete_doc(req: DeleteRequest):
    """Remove a document from both ChromaDB and SQLite."""
    try:
        delete_document(req.file_id)
        delete_document_record(req.file_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {e}")

    return DeleteResponse(file_id=req.file_id, message="Deleted successfully 🗑️")


# --- /chat-history -------------------------------------------------------
@app.get("/chat-history/{session_id}")
def chat_history(session_id: str):
    """Return chat history for a session as a list of {role, content} dicts."""
    return {"session_id": session_id, "messages": get_chat_history_dicts(session_id)}


# --- /clear-session ------------------------------------------------------
@app.delete("/clear-session/{session_id}")
def clear(session_id: str):
    """Delete a session's chat history, quizzes, and score."""
    clear_session(session_id)
    return {"session_id": session_id, "message": "Session cleared 🧹"}
