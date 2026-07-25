# Company Brain — Pro Edition

The full-featured version. One Streamlit app with a chat interface and an
admin dashboard, plus an optional Slack bot that shares the same brain.

## Feature checklist

| Feature | Where it lives |
|---|---|
| Highlight-the-source-passage | Chat tab, "View source passage(s)" expander under each answer |
| Confidence flagging (High/Medium/Low) | Badge shown under every answer, based on retrieval strength |
| Multi-document / ask across everything | Upload as many PDFs as you want; retrieval spans all of them |
| Contradiction detector | Built into the answer prompt — if retrieved excerpts from different docs disagree, the model is instructed to say so explicitly |
| Version diffing | Admin Dashboard → "Compare document versions" (tag uploads with a shared version group first) |
| Follow-up-aware chat | Last few turns are passed as context automatically |
| Saved Q&A library | Chat tab → "Save as approved FAQ answer" (Admin only); served instantly from cache next time, skipping the LLM call |
| Slack bot mode | `slack_bot.py` — separate script, needs your own Slack app (see setup notes inside the file) |
| Escalate-to-human | "🚩 Escalate to a human" link under every answer, opens a pre-filled email |
| Auto-generated FAQ | Admin Dashboard, built from real question frequency, no manual curation |
| Usage analytics | Admin Dashboard — total questions, low-confidence rate |
| Access control per document | Tag each upload with a role (All/Employee/HR/Finance/Legal/Admin) when uploading; the chat only retrieves from documents the current role can see |
| Audit log | Admin Dashboard — who asked what, when, with what confidence |

## 1. Install

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Add your Google Gemini API key

Copy `.env.example` to `.env` and fill in:

```
GEMINI_API_KEY=your-gemini-api-key-here
```

Get a key at https://aistudio.google.com/apikey. Or paste it into the sidebar once running.

This app calls Gemini's `gemini-2.5-flash` model — cheap and fast, well suited
to this kind of retrieval-augmented Q&A. Swap the `MODEL` constant at the top
of `rag_engine.py` if you want a different Gemini model.

## 3. Run the web app

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. The sidebar lets you pick a name and a
**simulated role** (there's no real login here — swap that dropdown for your
actual auth/SSO system before using this with real sensitive documents).

## 4. (Optional) Run the Slack bot

`slack_bot.py` reuses the exact same index as the web app — anything
ingested through one is visible to the other. It needs its own Slack app
credentials; full setup steps are in the docstring at the top of the file.

```bash
pip install slack_bolt
python slack_bot.py
```

## How it's built

- **`rag_engine.py`** — the core. Chunks PDFs by page/paragraph, stores them
  in a **persistent** ChromaDB collection (`./chroma_db/`, survives
  restarts), retrieves by semantic similarity filtered by access role,
  computes a confidence score from retrieval distance, and asks Claude to
  answer strictly from context — with instructions to flag disagreement
  across documents. Also handles version-comparison mode.
- **`storage.py`** — a small SQLite database (`company_brain.db`) tracking
  documents, the audit log, the saved-Q&A cache, and question frequency for
  the auto-FAQ. No external database server needed.
- **`app.py`** — the Streamlit UI: chat tab (with source highlighting,
  confidence badges, escalation links, save-to-FAQ) and an admin dashboard
  (analytics, audit log, FAQ, saved answers, version comparison, access
  control table).
- **`slack_bot.py`** — optional, separate process, same brain.

## Honest limitations (worth knowing before a real rollout)

- **Role-based access here is simulated**, not real auth. Anyone can pick
  "Admin" from a dropdown. Wire the role selector to your actual identity
  provider before this touches real confidential documents.
- **The FAQ cache match is near-exact text matching**, not semantic — "What's
  the return policy?" and "how do returns work?" won't be treated as the same
  question. A production version would want embedding-based matching here.
- **Confidence scoring is a heuristic** based on retrieval distance
  thresholds, not a calibrated probability. Treat "Low confidence" as "this
  deserves a second look," not a precise statistic.
- **Contradiction detection relies on the model noticing** disagreement
  across retrieved excerpts — it's prompted for, not a separate verified
  pass. For anything high-stakes, review flagged contradictions yourself.
- **SQLite + local ChromaDB** are fine for a single-server demo or small
  team tool; a larger rollout would want a proper hosted vector DB and
  database instead of local files.
