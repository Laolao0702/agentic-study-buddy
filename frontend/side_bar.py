"""
Sidebar UI for StudyBuddy
- File uploader
- List of indexed documents (with delete buttons)
- Clear session button
"""

import streamlit as st

from frontend.api_client import (
    upload_doc,
    list_docs,
    delete_doc,
    clear_session,
    get_quiz_score,
)


def render_sidebar():
    """Renders the entire sidebar. Called from app.py."""

    with st.sidebar:
        st.title("📚 StudyBuddy")
        st.caption("Your agentic AI study assistant")
        st.divider()

        # ============================================================
        # 1. Upload section
        # ============================================================
        st.subheader("📤 Upload a document")
        uploaded_file = st.file_uploader(
            "Choose a PDF or DOCX",
            type=["pdf", "docx"],
            key="file_uploader",
        )

        if uploaded_file is not None:
            if st.button("Upload & Index", type="primary", use_container_width=True):
                with st.spinner(f"Indexing {uploaded_file.name}..."):
                    try:
                        result = upload_doc(uploaded_file)
                        st.success(f"✅ {result['message']}")
                        st.rerun()  # refresh the doc list
                    except Exception as e:
                        st.error(f"Upload failed: {e}")

        st.divider()

        # ============================================================
        # 2. Documents list
        # ============================================================
        st.subheader("📚 Your Documents")

        try:
            docs = list_docs()
        except Exception as e:
            st.error(f"Couldn't load documents: {e}")
            docs = []

        if not docs:
            st.info("No documents yet. Upload one above to get started!")
        else:
            for doc in docs:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"📄 **{doc['filename']}**")
                    st.caption(f"Uploaded: {doc['uploaded_at'][:19]}")
                with col2:
                    if st.button("🗑️", key=f"del_{doc['id']}"):
                        try:
                            delete_doc(doc["id"])
                            st.success("Deleted")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Delete failed: {e}")

        st.divider()

        # ============================================================
        # 3. Quiz score
        # ============================================================
        sid = st.session_state.get("session_id")
        if sid:
            st.subheader("🎯 Quiz score")
            try:
                score = get_quiz_score(sid)
            except Exception:
                score = None

            if score and score.get("total"):
                correct, total = score["correct"], score["total"]
                st.metric("Correct answers", f"{correct}/{total}")
                st.progress(correct / total)
            else:
                st.caption("No quizzes taken yet. Try “Quiz me on …”")

            st.divider()

        # ============================================================
        # 4. Session controls
        # ============================================================
        st.subheader("🧹 Session")
        st.caption(f"ID: `{sid[:8] if sid else 'not started'}...`")

        if st.button("Clear chat history", use_container_width=True):
            session_id = st.session_state.get("session_id")
            if session_id:
                try:
                    clear_session(session_id)
                    st.session_state["messages"] = []
                    st.session_state["session_id"] = None
                    st.query_params.clear()
                    st.success("Chat cleared!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Clear failed: {e}")
            else:
                st.info("No session to clear yet.")