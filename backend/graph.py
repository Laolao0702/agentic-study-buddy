"""
The StudyBuddy agent graph.

    START -> route_agent -> run_tool -> format_response -> END

route_agent picks one tool, run_tool executes it, format_response turns the
tool's structured output into the message the user actually sees.
"""

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

from backend.config import CHAT_MODEL, require_api_key
from backend.state import Studybuddy
from backend.prompt import AGENT_SYSTEM_PROMPT
from backend.tool import (
    answer_from_docs,
    generate_quiz,
    summarize_topic,
    explain_concept,
    as_text,
)

require_api_key()

TOOL_REGISTRY = {
    "answer_from_docs": answer_from_docs,
    "generate_quiz": generate_quiz,
    "summarize_topic": summarize_topic,
    "explain_concept": explain_concept,
}

router_llm = ChatGoogleGenerativeAI(
    model=CHAT_MODEL,
    temperature=0,
    max_retries=3,
).bind_tools(list(TOOL_REGISTRY.values()))


# --- NODE: route_agent ---------------------------------------------------
def route_agent(state: Studybuddy) -> dict:
    """Ask the LLM which tool matches the user's intent."""
    messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT)]
    # Only the last exchange, so the router prompt doesn't grow without bound.
    messages.extend(state.get("messages", [])[-2:])
    messages.append(HumanMessage(content=state["user_input"]))

    response = router_llm.invoke(messages)

    if response.tool_calls:
        call = response.tool_calls[0]
        print(f"Router picked: {call['name']} with args {call['args']}")
        return {"tool_call": call["name"], "tool_args": call["args"]}

    # No tool needed — a greeting or small talk. Pass the text straight through.
    print("Router answered directly (no tool)")
    return {"tool_call": "none", "tool_args": {}, "raw_output": as_text(response.content)}


# --- NODE: run_tool ------------------------------------------------------
def run_tool(state: Studybuddy) -> dict:
    """Execute whichever tool the router picked."""
    tool_name = state["tool_call"]

    if tool_name == "none":
        return {}

    if tool_name not in TOOL_REGISTRY:
        return {"raw_output": {"error": "I'm not sure which action to take. Could you rephrase?"}}

    print(f"run_tool: executing '{tool_name}'")
    try:
        result = TOOL_REGISTRY[tool_name].invoke(state.get("tool_args") or {})
    except Exception as e:
        # Surface the failure instead of silently returning an empty answer.
        print(f"Tool error in {tool_name}: {e}")
        return {"raw_output": {"error": f"The {tool_name} step failed: {e}"}}

    updates = {"raw_output": result}
    if tool_name == "generate_quiz" and isinstance(result, dict) and "error" not in result:
        updates["quiz_questions"] = result.get("questions", [])
    return updates


# --- NODE: format_response -----------------------------------------------
def format_response(state: Studybuddy) -> dict:
    """Turn the tool's output into the user-facing message."""
    tool_name = state["tool_call"]
    raw = state.get("raw_output")

    # Any tool can report a soft failure this way.
    if isinstance(raw, dict) and "error" in raw:
        final_ans = raw["error"]

    elif tool_name == "none":
        final_ans = raw if isinstance(raw, str) and raw.strip() else (
            "Hi! Ask me a question about your documents, or request a quiz, "
            "summary, or explanation."
        )

    elif tool_name == "answer_from_docs":
        final_ans = raw if isinstance(raw, str) else str(raw)

    elif tool_name == "generate_quiz" and isinstance(raw, dict):
        # Deliberately no questions or answers in the text: the UI renders an
        # interactive quiz from state["quiz_questions"] and grades it server-side.
        count = len(raw.get("questions", []))
        topic = raw.get("topic", "your documents")
        final_ans = (
            f"Here's a {count}-question quiz on **{topic}**. "
            "Pick your answers below and hit Submit."
        )

    elif tool_name == "summarize_topic" and isinstance(raw, dict):
        lines = [f"**{raw.get('title', 'Summary')}**\n"]
        lines += [f"- {b}" for b in raw.get("bullets", [])]
        final_ans = "\n".join(lines)

    elif tool_name == "explain_concept" and isinstance(raw, dict):
        final_ans = (
            f"**{raw.get('concept', '')}**\n\n"
            f"{raw.get('explanation', '')}\n\n"
            f"*Analogy:* {raw.get('analogy', '')}"
        )

    else:
        final_ans = str(raw)

    messages = list(state.get("messages", []))
    messages.append(HumanMessage(content=state["user_input"]))
    messages.append(AIMessage(content=final_ans))

    print(f"format_response: produced {len(final_ans)} chars")
    return {"final_ans": final_ans, "messages": messages}


# --- Graph ---------------------------------------------------------------
def build_graph():
    graph = StateGraph(Studybuddy)
    graph.add_node("route_agent", route_agent)
    graph.add_node("run_tool", run_tool)
    graph.add_node("format_response", format_response)

    graph.add_edge(START, "route_agent")
    graph.add_edge("route_agent", "run_tool")
    graph.add_edge("run_tool", "format_response")
    graph.add_edge("format_response", END)
    return graph.compile()


agent_graph = build_graph()


def run_agent(user_input: str, session_id: str = "default", history: list = None) -> dict:
    """Run the agent for a single user message and return the final state."""
    initial_state: Studybuddy = {
        "messages": history or [],
        "session_id": session_id,
        "user_input": user_input,
        "tool_call": "",
        "tool_args": {},
        "raw_output": None,
        "quiz_questions": [],
        "final_ans": "",
    }
    return agent_graph.invoke(initial_state)
