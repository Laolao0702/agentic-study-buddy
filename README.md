# StudyBuddy — Agentic RAG Study Assistant

Upload your course notes, then ask questions, get summaries and explanations,
or be quizzed on them — with every answer drawn from *your* documents rather
than the model's memory.

**Stack:** FastAPI · LangGraph · ChromaDB · Google Gemini · Streamlit · SQLite

---

## The problem

Asking a general chatbot about your coursework has two failure modes: it
answers confidently from training data that may contradict your syllabus, and
it can't quiz you on the specific material you actually have to sit an exam on.

Meanwhile, a plain RAG app solves grounding but not *intent* — "explain
photosynthesis simply", "summarize chapter 3", and "quiz me on chapter 3" want
genuinely different outputs, and stuffing all three into one prompt produces
mediocre versions of each.

StudyBuddy addresses both:

- **Grounding** — documents are chunked, embedded, and retrieved per query, so
  answers cite your uploaded material. Ask about something not in your notes
  and it says so rather than inventing an answer.
- **Routing** — a LangGraph agent classifies each message and dispatches to one
  of four purpose-built tools, each with its own prompt and its own structured
  output schema. One chat box, four specialised behaviours.
- **Active recall** — quizzes are interactive and graded server-side, so the
  browser never holds the answer key, and re-submitting a quiz can't inflate
  your score.

## Example

```
You:  Quiz me on photosynthesis with 2 questions
      → routes to generate_quiz → retrieves 5 chunks → returns MCQs
      → you answer in the UI → graded server-side → 1/2, with explanations

You:  Explain chlorophyll like I'm five
      → routes to explain_concept → plain-English explanation + analogy

You:  What caused the French Revolution?
      → routes to answer_from_docs → "I couldn't find the answer in the
        provided context."   (nothing in your notes covers it)
```

---

## Architecture

```
Streamlit UI  ──HTTP──>  FastAPI  ──>  LangGraph agent
 (frontend/)              (backend/main.py)   │
                                              ├─ route_agent      picks ONE tool
                                              ├─ run_tool         executes it
                                              └─ format_response  builds the reply
                                                      │
                                    ┌─────────────────┴──────────────────┐
                            Chroma (vectors)                  SQLite (rag_app.db)
                            gemini-embedding-001              chat logs, docs,
                                                              quizzes, scores
```

The agent picks exactly one of four tools per message:

| Tool | Triggered by | Returns |
|---|---|---|
| `answer_from_docs` | a factual question | grounded prose answer |
| `summarize_topic`  | "summarize", "overview" | title + bullet points |
| `explain_concept`  | "explain simply", "ELI5" | explanation + analogy |
| `generate_quiz`    | "quiz me", "test me" | interactive MCQs, graded server-side |

---

## Setup

Requires **Python 3.12** and a Google Gemini API key
([get one free](https://aistudio.google.com/apikey)).

```bash
cd AgenticStudyBuddy

python3.12 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env` in the project root:

```ini
GOOGLE_API_KEY=your_key_here

# Models — change here, nowhere else in the code
CHAT_MODEL=gemini-flash-lite-latest
EMBEDDING_MODEL=models/gemini-embedding-001

# Optional: LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=studybuddy-agent
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

## Running

Two terminals, both from the project root:

```bash
# Terminal 1 — backend
source venv/bin/activate
uvicorn backend.main:app --reload

# Terminal 2 — frontend
source venv/bin/activate
streamlit run frontend/app.py
```

Then open <http://localhost:8501>. API docs are at <http://localhost:8000/docs>.

Run the backend from the **project root**, not from `backend/` — the module
imports (`backend.graph`) and the default data paths both assume it.

---

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/chat` | send a message, run the agent |
| `POST` | `/upload-doc` | upload + index a PDF/DOCX |
| `GET`  | `/list-docs` | list indexed documents |
| `POST` | `/delete-doc` | remove a document and its vectors |
| `POST` | `/submit-quiz` | grade a quiz, update the running score |
| `GET`  | `/quiz-score/{session_id}` | running score for a session |
| `GET`  | `/chat-history/{session_id}` | rehydrate the UI after refresh |
| `DELETE` | `/clear-session/{session_id}` | wipe a session |

Quiz answer keys are stored server-side and never sent to the browser —
`/chat` returns questions and options only, and `/submit-quiz` does the grading.

---

## Gotchas

**Changing `EMBEDDING_MODEL` breaks the existing index.** The Chroma collection
is built at a fixed dimensionality (3072 for `gemini-embedding-001`). If you
switch embedding models you must delete `backend/chroma_db/` and re-upload
every document.

**Model names change.** `CHAT_MODEL` is a `-latest` alias on purpose: this
project originally hardcoded a Groq model that was retired, which broke every
LLM call at once. All model names now live in `backend/config.py`.

**Free-tier quotas vary a lot by model.** `gemini-3.8-flash` allows only 20
requests/day on the free tier; `gemini-flash-lite-latest` is far more generous
and routed all four tools correctly in testing. To see what your key can use:

```bash
curl -s "https://generativelanguage.googleapis.com/v1beta/models?key=$GOOGLE_API_KEY" \
  | python3 -c "import sys,json;[print(m['name']) for m in json.load(sys.stdin)['models']]"
```

**Gemini 3.x returns list-shaped content**, not plain strings. `backend/tool.py`
has an `as_text()` helper for this; use it on any new `llm.invoke(...).content`
you add, or raw dicts will leak into the chat UI.

---

## Project layout

```
backend/
  config.py    model names + paths — the single place to change a model
  main.py      FastAPI app, 9 endpoints
  graph.py     LangGraph agent: route_agent → run_tool → format_response
  tool.py      the four tools + Pydantic schemas for structured output
  rag.py       chunking, embedding, Chroma retrieval
  prompt.py    prompt templates
  db.py        SQLite: chat logs, documents, quizzes, scores
  model.py     Pydantic request/response models
frontend/
  app.py       Streamlit entry point
  chat_ui.py   chat rendering + interactive quiz widget
  side_bar.py  upload, document list, quiz score
  api_client.py HTTP wrapper around the backend
```

## Roadmap

- [ ] Relevance threshold on retrieval — `similarity_search` always returns the
      *k* nearest chunks, so "nothing relevant found" is currently only
      detectable for `answer_from_docs` (via its prompt). `generate_quiz` will
      quiz you on whatever came back, even for an off-topic request.
- [ ] Per-document filtering, so you can scope a quiz to one upload
- [ ] Export quiz results for revision tracking

## Credits

Structure and approach follow the
[FutureSmart AI LangChain RAG course](https://blog.futuresmart.ai/series/langchain-rag-course),
extended with LangGraph routing, interactive server-side-graded quizzes,
and session scoring.
