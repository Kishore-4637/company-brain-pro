"""
Company Brain — full-featured edition.

Features:
- Multi-document ingestion with per-document access roles
- Source-passage highlighting under every answer (click to verify)
- Confidence flagging (High/Medium/Low) based on retrieval strength
- Follow-up-aware chat (short conversational memory)
- Saved Q&A cache (admin-approved instant answers)
- Auto-generated FAQ from real usage
- Contradiction awareness across multiple documents
- Version comparison tool ("what changed between v1 and v2")
- Escalate-to-human button (pre-filled email)
- Usage analytics + audit log (admin dashboard)
- Simulated role-based access control

Run with:  streamlit run app.py
"""

import os
import urllib.parse

import streamlit as st
from dotenv import load_dotenv

import storage
from rag_engine import CompanyBrain

load_dotenv()

st.set_page_config(page_title="Company Brain", page_icon="🧠", layout="wide")

ROLES = ["All", "Employee", "HR", "Finance", "Legal", "Admin"]


def get_api_key() -> str:
    return st.session_state.get("api_key") or os.getenv("GEMINI_API_KEY", "")


def confidence_badge(level: str) -> str:
    return {"High": "🟢 High confidence", "Medium": "🟡 Medium confidence", "Low": "🔴 Low confidence"}.get(level, level)


if "brain" not in st.session_state:
    st.session_state["brain"] = CompanyBrain()
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []  # list[(role, text)]
if "last_sources" not in st.session_state:
    st.session_state["last_sources"] = {}  # msg_index -> list[SourceChunk]

brain: CompanyBrain = st.session_state["brain"]

# ===================== Sidebar =====================
with st.sidebar:
    st.header("🧠 Company Brain")

    st.subheader("Who's asking?")
    user_name = st.text_input("Your name", value=st.session_state.get("user_name", "Guest"))
    st.session_state["user_name"] = user_name
    current_role = st.selectbox("Your role", ROLES, index=ROLES.index(st.session_state.get("current_role", "Admin")))
    st.session_state["current_role"] = current_role
    st.caption("Simulated permissions — a real deployment would tie this to your actual login/SSO.")

    st.divider()
    st.subheader("Setup")
    api_key_input = st.text_input(
        "Google Gemini API key", value=os.getenv("GEMINI_API_KEY", ""), type="password",
        help="Reads from .env by default. Paste here for this session only.",
    )
    st.session_state["api_key"] = api_key_input

    st.divider()
    st.subheader("Upload a document")
    up_file = st.file_uploader("PDF", type=["pdf"], key="uploader")
    up_role = st.selectbox("Who can see this document?", ROLES, index=0, key="upload_role")

    with st.expander("Mark as a version (for version comparison)"):
        version_group = st.text_input("Version group name", placeholder="e.g. 'employee-handbook'")
        version_label = st.text_input("This version's label", placeholder="e.g. '2025 edition'")

    if st.button("Ingest document", use_container_width=True, disabled=up_file is None):
        with st.spinner("Reading and indexing..."):
            doc_id, n_chunks = brain.ingest(
                up_file.getvalue(), up_file.name, up_role,
                version_group=version_group or None, version_label=version_label or None,
            )
        st.success(f"Indexed '{up_file.name}' — {n_chunks} chunks ({up_role} access)")
        st.rerun()

    docs = brain.documents()
    if docs:
        st.caption(f"{len(docs)} document(s) indexed:")
        for d in docs:
            tag = f" · v: {d['version_label']}" if d["version_label"] else ""
            st.caption(f"📄 {d['filename']} — *{d['access_role']}*{tag}")
        if st.button("🗑️ Clear all documents", use_container_width=True):
            brain.clear_all()
            st.session_state["chat_history"] = []
            st.rerun()
    else:
        st.caption("No documents indexed yet.")

# ===================== Main =====================
st.title("Company Brain")
st.caption("Ask a document anything. Answers are grounded, cited, and flagged when confidence is low.")

tab_chat, tab_admin = st.tabs(["💬 Chat", "📊 Admin Dashboard"])

# --------------------------------------------------------------------------
with tab_chat:
    if not docs:
        st.info("👈 Upload a PDF in the sidebar to get started.")
    else:
        for i, (role, msg) in enumerate(st.session_state["chat_history"]):
            with st.chat_message(role):
                st.markdown(msg)
                if role == "assistant" and i in st.session_state["last_sources"]:
                    info = st.session_state["last_sources"][i]
                    c1, c2 = st.columns([1, 3])
                    with c1:
                        st.caption(confidence_badge(info["confidence"]))
                    with c2:
                        subject = urllib.parse.quote(f"Question needs human review: {info['question'][:60]}")
                        body = urllib.parse.quote(
                            f"Question: {info['question']}\n\nAI answer:\n{msg}\n\nConfidence: {info['confidence']}"
                        )
                        st.markdown(f"[🚩 Escalate to a human](mailto:support@yourcompany.com?subject={subject}&body={body})")

                    if info["sources"]:
                        with st.expander(f"📎 View {len(info['sources'])} source passage(s)"):
                            for s in info["sources"]:
                                st.markdown(
                                    f"**{s.doc_name} — Page {s.page}**  \n"
                                    f"<div style='background-color:#fff3b0;padding:8px;border-radius:4px;color:#111;'>{s.text}</div>",
                                    unsafe_allow_html=True,
                                )
                    if current_role == "Admin" and not info.get("cached"):
                        if st.button("💾 Save as approved FAQ answer", key=f"save_{i}"):
                            storage.save_qa(info["question"], msg, user_name)
                            st.success("Saved — future matching questions will use this answer instantly.")

        question = st.chat_input("Ask something about your documents...")
        if question:
            st.session_state["chat_history"].append(("user", question))
            with st.chat_message("user"):
                st.markdown(question)

            allowed_roles = ["All"] if current_role == "All" else ["All", current_role]

            with st.chat_message("assistant"):
                if not get_api_key():
                    answer_text = "⚠️ Please add your Google Gemini API key in the sidebar first."
                    st.markdown(answer_text)
                    confidence, sources, cached = "Low", [], False
                else:
                    with st.spinner("Searching documents..."):
                        result = brain.ask(
                            question, get_api_key(), allowed_roles,
                            chat_history=st.session_state["chat_history"][:-1],
                            user=user_name,
                        )
                    answer_text = result.text
                    confidence, sources, cached = result.confidence, result.sources, result.from_cache
                    st.markdown(answer_text)
                    if cached:
                        st.caption("⚡ Served from the approved FAQ cache")

            msg_index = len(st.session_state["chat_history"])
            st.session_state["chat_history"].append(("assistant", answer_text))
            st.session_state["last_sources"][msg_index] = {
                "question": question, "confidence": confidence, "sources": sources, "cached": cached,
            }
            st.rerun()

# --------------------------------------------------------------------------
with tab_admin:
    if current_role != "Admin":
        st.warning("This dashboard is only visible to the Admin role. Switch roles in the sidebar to preview it.")
    else:
        stats = storage.audit_stats()
        c1, c2, c3 = st.columns(3)
        c1.metric("Questions asked", stats["total_questions"])
        low_rate = f"{(stats['low_confidence'] / stats['total_questions'] * 100):.0f}%" if stats["total_questions"] else "—"
        c2.metric("Low-confidence answers", stats["low_confidence"], help="Answers where retrieval was weak — often signals a gap in the source document.")
        c3.metric("Low-confidence rate", low_rate)

        st.divider()
        left, right = st.columns(2)

        with left:
            st.subheader("🔥 Auto-generated FAQ")
            st.caption("Built from real questions people have actually asked.")
            top_qs = storage.top_questions(10)
            if top_qs:
                for q in top_qs:
                    st.markdown(f"- **{q['display_question']}** _(asked {q['count']}×)_")
            else:
                st.caption("No questions asked yet.")

        with right:
            st.subheader("✅ Saved / approved answers")
            saved = storage.list_saved_qa()
            if saved:
                for qa in saved:
                    with st.expander(qa["question"][:80]):
                        st.write(qa["answer"])
                        st.caption(f"Approved by {qa['created_by']}")
                        if st.button("Delete", key=f"del_qa_{qa['id']}"):
                            storage.delete_saved_qa(qa["id"])
                            st.rerun()
            else:
                st.caption("No approved answers yet — save one from the Chat tab.")

        st.divider()
        st.subheader("🔀 Compare document versions")
        version_groups = sorted({d["version_group"] for d in docs if d["version_group"]})
        if not version_groups:
            st.caption("Tag two uploads with the same 'version group' name in the sidebar to unlock this.")
        else:
            vg = st.selectbox("Version group", version_groups)
            candidates = [d for d in docs if d["version_group"] == vg]
            if len(candidates) < 2:
                st.caption("Upload at least two documents in this version group to compare them.")
            else:
                names = {f"{d['filename']} ({d['version_label'] or 'unlabeled'})": d["id"] for d in candidates}
                c1, c2 = st.columns(2)
                choice_a = c1.selectbox("Version A", list(names.keys()), index=0)
                choice_b = c2.selectbox("Version B", list(names.keys()), index=min(1, len(names) - 1))
                compare_q = st.text_input("What should we compare?", placeholder="e.g. the return policy")
                if st.button("Compare", disabled=not compare_q):
                    if not get_api_key():
                        st.warning("Add your API key in the sidebar first.")
                    else:
                        with st.spinner("Comparing..."):
                            diff = brain.compare_versions(
                                compare_q, names[choice_a], names[choice_b], get_api_key(),
                                label_a=choice_a, label_b=choice_b,
                            )
                        st.markdown(diff)

        st.divider()
        st.subheader("📜 Audit log")
        log = storage.get_audit_log(limit=50)
        if log:
            st.dataframe(
                [{"When": __import__("datetime").datetime.fromtimestamp(r["timestamp"]).strftime("%Y-%m-%d %H:%M"),
                  "User": r["user"], "Role": r["role"], "Question": r["question"],
                  "Confidence": r["confidence"]} for r in log],
                use_container_width=True, hide_index=True,
            )
        else:
            st.caption("No activity logged yet.")

        st.divider()
        st.subheader("📁 Document access control")
        if docs:
            st.dataframe(
                [{"Document": d["filename"], "Access role": d["access_role"],
                  "Version group": d["version_group"] or "—", "Chunks": d["chunk_count"]} for d in docs],
                use_container_width=True, hide_index=True,
            )
        else:
            st.caption("No documents uploaded yet.")
