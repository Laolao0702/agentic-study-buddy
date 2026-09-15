from typing import TypedDict, List, Dict, Any

from langchain_core.messages import BaseMessage


class Studybuddy(TypedDict):
    """State passed between graph nodes."""

    messages: List[BaseMessage]
    session_id: str
    user_input: str

    # Which tool the router chose, and the arguments it extracted.
    tool_call: str
    tool_args: Dict[str, Any]

    # Whatever the tool returned, before formatting.
    raw_output: Any

    # Populated only by generate_quiz, so the UI can render an interactive quiz.
    quiz_questions: List[Dict[str, Any]]

    final_ans: str
