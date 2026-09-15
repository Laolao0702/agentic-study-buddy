"""
The four StudyBuddy tools the router can choose between.

Each tool retrieves from the user's documents, then asks the LLM to shape that
context into a specific output. Tools raise no exceptions for the "nothing
found" case — they return an {"error": ...} dict the graph turns into a
friendly message.
"""

from typing import List

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from backend.config import CHAT_MODEL
from backend.rag import retrieve_docs, has_documents
from backend.prompt import (
    ANSWER_PROMPT,
    QUIZ_PROMPT,
    SUMMARY_PROMPT,
    EXPLAIN_PROMPT,
)

# max_retries smooths over the 503 "high demand" blips Gemini returns under load.
llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0.3, max_retries=3)


def combine_chunks(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def as_text(content) -> str:
    """
    Flatten a message's content to a plain string.

    Gemini 3.x returns content as a list of blocks —
    [{"type": "text", "text": "..."}] — not a bare string. Rendering that
    straight into the UI leaks raw Python dicts into the chat.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts).strip()
    return str(content)


def _no_docs_message(subject: str) -> str:
    """Distinguish 'you uploaded nothing' from 'nothing matched'."""
    if not has_documents():
        return "You haven't uploaded any documents yet. Add a PDF or DOCX in the sidebar and I'll study it with you."
    return f"I couldn't find anything about “{subject}” in your uploaded documents."


# --- Structured output schemas ------------------------------------------
class QuizQuestion(BaseModel):
    """A single multiple-choice question."""
    question: str = Field(description="The question text")
    options: List[str] = Field(description="Exactly 4 answer options, in order A, B, C, D")
    correct: str = Field(description="The letter of the correct option: A, B, C, or D")
    explanation: str = Field(description="One sentence explaining why that answer is correct")


class Quiz(BaseModel):
    """A collection of quiz questions."""
    questions: List[QuizQuestion] = Field(description="The list of quiz questions")


class Summary(BaseModel):
    """A structured summary of a topic."""
    title: str = Field(description="A short title for the summary")
    bullets: List[str] = Field(description="Key points as concise bullet points")


class Explanation(BaseModel):
    """A clear, beginner-friendly explanation of a concept."""
    concept: str = Field(description="The concept being explained")
    explanation: str = Field(description="A clear, plain-English explanation")
    analogy: str = Field(description="An everyday analogy to help understand the concept")


# --- Tools ---------------------------------------------------------------
@tool
def answer_from_docs(query: str) -> str:
    """
    Answer a user's question using their uploaded study materials.
    Use this when the user asks a factual question that should be
    grounded in their documents (e.g., 'What is photosynthesis?',
    'Explain mitosis according to my textbook').
    """
    docs = retrieve_docs(query, k=5)
    if not docs:
        return _no_docs_message(query)
    messages = ANSWER_PROMPT.format_messages(
        context=combine_chunks(docs), question=query
    )
    return as_text(llm.invoke(messages).content)


@tool
def generate_quiz(topic: str, num_questions: int = 3) -> dict:
    """
    Generate multiple-choice quiz questions on a topic from the user's
    uploaded documents. Use this when the user asks to be quizzed or
    tested on something (e.g., 'Quiz me on cells', 'Make 5 MCQs about
    photosynthesis').
    """
    docs = retrieve_docs(topic, k=5)
    if not docs:
        return {"error": _no_docs_message(topic)}

    num_questions = max(1, min(int(num_questions or 3), 10))
    messages = QUIZ_PROMPT.format_messages(
        context=combine_chunks(docs), topic=topic, num_questions=num_questions
    )
    quiz = llm.with_structured_output(Quiz).invoke(messages)
    result = quiz.model_dump()
    result["topic"] = topic
    return result


@tool
def summarize_topic(topic: str) -> dict:
    """
    Produce a structured summary of a topic from the user's uploaded
    documents. Use this when the user asks for a summary, overview,
    or recap (e.g., 'Summarize chapter 3', 'Give me an overview of cells').
    """
    docs = retrieve_docs(topic, k=6)
    if not docs:
        return {"error": _no_docs_message(topic)}

    messages = SUMMARY_PROMPT.format_messages(
        context=combine_chunks(docs), topic=topic
    )
    return llm.with_structured_output(Summary).invoke(messages).model_dump()


@tool
def explain_concept(concept: str) -> dict:
    """
    Explain a concept in plain English using the user's uploaded documents.
    Use this when the user wants something explained simply (e.g.,
    'Explain DNA like I'm 5', 'What does mitochondria mean in simple terms').
    """
    docs = retrieve_docs(concept, k=4)
    if not docs:
        return {"error": _no_docs_message(concept)}

    messages = EXPLAIN_PROMPT.format_messages(
        context=combine_chunks(docs), concept=concept
    )
    return llm.with_structured_output(Explanation).invoke(messages).model_dump()
