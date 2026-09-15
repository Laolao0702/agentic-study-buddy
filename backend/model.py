"""
Pydantic schemas for FastAPI request/response validation.
"""

from typing import List, Optional, Dict

from pydantic import BaseModel, Field


# --- /chat ---------------------------------------------------------------
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's message")
    session_id: Optional[str] = Field(None, description="Session ID; created if missing")


class QuizQuestionPublic(BaseModel):
    """A quiz question as sent to the browser — no `correct` field."""
    index: int
    question: str
    options: List[str]


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    tool_used: str
    quiz_id: Optional[int] = None
    quiz_questions: Optional[List[QuizQuestionPublic]] = None


# --- /upload-doc ---------------------------------------------------------
class UploadResponse(BaseModel):
    file_id: int
    filename: str
    message: str


# --- /list-docs ----------------------------------------------------------
class DocumentInfo(BaseModel):
    id: int
    filename: str
    uploaded_at: str


# --- /delete-doc ---------------------------------------------------------
class DeleteRequest(BaseModel):
    file_id: int


class DeleteResponse(BaseModel):
    file_id: int
    message: str


# --- /submit-quiz --------------------------------------------------------
class QuizSubmission(BaseModel):
    quiz_id: int
    answers: Dict[int, str] = Field(
        ..., description="Map of question index -> chosen letter (A/B/C/D)"
    )


class GradedAnswer(BaseModel):
    index: int
    question: str
    chosen: Optional[str]
    correct: str
    is_correct: bool
    explanation: str = ""


class QuizResult(BaseModel):
    quiz_id: int
    correct: int
    total: int
    results: List[GradedAnswer]
    session_score: Dict[str, int]


# --- /quiz-score ---------------------------------------------------------
class ScoreResponse(BaseModel):
    session_id: str
    correct: int
    total: int
