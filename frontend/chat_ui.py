"""
Chat interface for StudyBuddy
- Renders chat history
- Handles new messages
- Renders interactive quizzes and grades them via the backend
"""

import streamlit as st

from frontend.api_client import send_message, submit_quiz

LETTERS = ["A", "B", "C", "D", "E", "F"]


def _render_quiz(msg: dict, msg_index: int):
    """Render one quiz: radio buttons before grading, results after."""
    quiz = msg["quiz"]
    quiz_id = quiz["quiz_id"]
    questions = quiz["questions"]
    graded = msg.get("graded")

    if graded:
        _render_quiz_results(graded)
        return

    with st.form(key=f"quiz_form_{quiz_id}"):
        choices = {}
        for q in questions:
            idx = q["index"]
            labels = [
                f"{LETTERS[i]}) {opt}" for i, opt in enumerate(q["options"])
            ]
            picked = st.radio(
                f"**Q{idx + 1}. {q['question']}**",
                options=labels,
                key=f"quiz_{quiz_id}_q{idx}",
                index=None,
            )
            # Map the display label back to its letter.
            choices[idx] = LETTERS[labels.index(picked)] if picked else ""

        submitted = st.form_submit_button("Submit answers", type="primary")

    if submitted:
        unanswered = [i for i, v in choices.items() if not v]
        if unanswered:
            st.warning(
                "Please answer every question first "
                f"(missing: {', '.join(f'Q{i + 1}' for i in unanswered)})."
            )
            return
        try:
            result = submit_quiz(quiz_id, choices)
        except Exception as e:
            st.error(f"Couldn't grade the quiz: {e}")
            return

        st.session_state["messages"][msg_index]["graded"] = result
        st.session_state["quiz_score"] = result.get("session_score")
        st.rerun()


def _render_quiz_results(graded: dict):
    """Show the score and a per-question breakdown."""
    correct, total = graded["correct"], graded["total"]
    pct = (correct / total * 100) if total else 0

    if pct == 100:
        st.success(f"🎉 Perfect — {correct}/{total}")
    elif pct >= 50:
        st.info(f"You scored {correct}/{total}")
    else:
        st.warning(f"You scored {correct}/{total} — worth another look")

    for r in graded["results"]:
        icon = "✅" if r["is_correct"] else "❌"
        with st.expander(f"{icon} Q{r['index'] + 1}. {r['question']}", expanded=not r["is_correct"]):
            if r["is_correct"]:
                st.markdown(f"Your answer: **{r['chosen']}** — correct.")
            else:
                st.markdown(
                    f"Your answer: **{r['chosen'] or '—'}**  \n"
                    f"Correct answer: **{r['correct']}**"
                )
            if r.get("explanation"):
                st.caption(r["explanation"])

    score = graded.get("session_score") or {}
    if score.get("total"):
        st.caption(
            f"📊 Session total: {score['correct']}/{score['total']} "
            f"({score['correct'] / score['total'] * 100:.0f}%)"
        )


def render_chat():
    """Renders the chat UI in the main area. Called from app.py."""

    st.title("💬 Chat with StudyBuddy")
    st.caption("Ask questions, request quizzes, summaries, or explanations of concepts.")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # --- Render the conversation so far ---------------------------------
    for i, msg in enumerate(st.session_state["messages"]):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("quiz"):
                _render_quiz(msg, i)
            if msg.get("tool_used"):
                st.caption(f"🛠️ Tool used: `{msg['tool_used']}`")

    # --- Handle new input ------------------------------------------------
    if user_input := st.chat_input("Ask me anything about your study materials..."):
        st.session_state["messages"].append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"), st.spinner("Thinking..."):
            try:
                result = send_message(
                    question=user_input,
                    session_id=st.session_state.get("session_id"),
                )
            except Exception as e:
                st.error(f"Backend error: {e}")
                st.session_state["messages"].pop()  # don't strand the user turn
                return

        # FastAPI may have created the session; mirror it into the URL so a
        # refresh can restore the chat.
        new_sid = result["session_id"]
        if new_sid != st.session_state.get("session_id"):
            st.session_state["session_id"] = new_sid
            st.query_params["sid"] = new_sid

        assistant_msg = {
            "role": "assistant",
            "content": result.get("answer", "(no response)"),
            "tool_used": result.get("tool_used", "unknown"),
        }
        if result.get("quiz_id") and result.get("quiz_questions"):
            assistant_msg["quiz"] = {
                "quiz_id": result["quiz_id"],
                "questions": result["quiz_questions"],
            }

        st.session_state["messages"].append(assistant_msg)
        # Rerun so the message renders through the normal loop above — that's
        # what makes the quiz widgets survive later reruns.
        st.rerun()
