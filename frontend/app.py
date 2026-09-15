"""
StudyBuddy — Streamlit entry point
Run with: streamlit run frontend/app.py
"""

import sys, os
# Put the project root on the path so `frontend.*` imports resolve even when
# Streamlit is launched as `streamlit run frontend/app.py` (which otherwise only
# adds the frontend/ folder to sys.path, not the project root).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from frontend.side_bar import render_sidebar
from frontend.chat_ui import render_chat
from frontend.api_client import get_chat_history


# ============================================================
# Page configuration
# ============================================================
st.set_page_config(
    page_title="StudyBuddy Agent",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Initialize session state once
# ============================================================
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "session_id" not in st.session_state:
    st.session_state["session_id"] = None  # FastAPI will create one on first /chat


# ============================================================
# Hydrate from URL query param so refresh preserves the chat
# ============================================================
if st.session_state["session_id"] is None:
    url_sid = st.query_params.get("sid")
    if url_sid:
        try:
            data = get_chat_history(url_sid)
            st.session_state["session_id"] = url_sid
            st.session_state["messages"] = data.get("messages", [])
        except Exception as e:
            st.warning(f"Couldn't restore previous session: {e}")
            st.query_params.clear()


# ============================================================
# Render the app
# ============================================================
render_sidebar()
render_chat()