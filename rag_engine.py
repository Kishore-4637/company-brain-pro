"""
rag_engine.py — the core of Company Brain.

Handles:
- ingesting one or more PDFs into a persistent, shared ChromaDB index
- access-role tagging per document (simple simulated permissions)
- retrieval-augmented answering with:
    - page + document citations
    - confidence scoring (so the app can flag weak answers)
    - source passage text returned for highlighting in the UI
    - awareness of multiple source documents (flags disagreement)
    - short conversational memory for natural follow-up questions
- a version-comparison mode for "what changed between doc A and doc B"
"""

import io
import re
import uuid
from dataclasses import dataclass, field

from google import genai
import chromadb
import pdfplumber

import storage

MODEL = "gemini-2.5-flash"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "company_docs"

ANSWER_PROMPT = """You are answering a question using ONLY the context excerpts below,
pulled from one or more company documents. Each excerpt is labeled with the
document it came from and the page number.

Rules:
- Answer clearly and concisely, using only what's in the context.
- Always cite your source(s) at the end like this: [Found in {{document}}, Page {{n}}].
  If multiple excerpts support the answer, cite all of them.
- If excerpts from DIFFERENT documents disagree or say different things about
  the same topic, point that out explicitly rather than picking one silently.
- If the context does not contain the answer, say plainly that the provided
  document(s) don't appear to cover it. Do not make anything up.
{history_block}
Context:
{context}

Question: {question}
"""

COMPARE_PROMPT = """Compare how these two versions of a document handle the
topic below. Excerpts from Version A and Version B are provided. Summarize
what changed (or confirm nothing relevant changed), in plain English.

Topic / question: {question}

--- Version A ({label_a}) ---
{context_a}

--- Version B ({label_b}) ---
{context_b}
"""


@dataclass
class SourceChunk:
    text: str
    doc_name: str
    page: int
    distance: float


@dataclass
class Answer:
    text: str
    confidence: str  # "High" | "Medium" | "Low"
    sources: list = field(default_factory=list)  # list[SourceChunk]
    from_cache: bool = False


class CompanyBrain:
    """Wraps a persistent Chroma collection shared across all ingested documents."""

    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.client.get_or_create_collection(COLLECTION_NAME)
        storage.init_db()

    # ---------------- ingestion ----------------

    def ingest(self, file_bytes: bytes, filename: str, access_role: str = "All",
               version_group: str | None = None, version_label: str | None = None) -> tuple[str, int]:
        """Chunk a PDF by page/paragraph and add it to the shared index.
        Returns (doc_id, chunk_count)."""
        doc_id = uuid.uuid4().hex[:12]
        chunks = _extract_chunks(file_bytes)

        if chunks:
            self.collection.add(
                documents=[c.text for c in chunks],
                metadatas=[{
                    "doc_id": doc_id,
                    "doc_name": filename,
                    "page": c.page,
                    "access_role": access_role,
                } for c in chunks],
                ids=[f"{doc_id}_{i}" for i in range(len(chunks))],
            )

        storage.add_document(doc_id, filename, access_role, len(chunks), version_group, version_label)
        return doc_id, len(chunks)

    def documents(self) -> list[dict]:
        return storage.list_documents()

    def clear_all(self):
        try:
            self.client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(COLLECTION_NAME)
        storage.clear_documents()

    # ---------------- retrieval + answering ----------------

    def _retrieve(self, question: str, allowed_roles: list[str], top_k: int,
                   doc_id_filter: str | None = None) -> list[SourceChunk]:
        where_clause = None
        if doc_id_filter:
            where_clause = {"doc_id": doc_id_filter}
        elif allowed_roles:
            where_clause = {"access_role": {"$in": allowed_roles}}

        count = self.collection.count()
        if count == 0:
            return []

        results = self.collection.query(
            query_texts=[question],
            n_results=min(top_k, count),
            where=where_clause,
        )
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        return [
            SourceChunk(text=d, doc_name=m["doc_name"], page=m["page"], distance=dist)
            for d, m, dist in zip(docs, metas, dists)
        ]

    @staticmethod
    def _confidence_from(sources: list[SourceChunk]) -> str:
        if not sources:
            return "Low"
        best = min(s.distance for s in sources)
        # Chroma's default embedding uses cosine distance in roughly 0-2 range.
        # These thresholds are a practical heuristic, not a calibrated metric.
        if best < 0.55:
            return "High"
        if best < 0.9:
            return "Medium"
        return "Low"

    def ask(self, question: str, api_key: str, allowed_roles: list[str],
            chat_history: list[tuple[str, str]] | None = None, top_k: int = 6,
            user: str = "anonymous", use_cache: bool = True) -> Answer:

        if self.collection.count() == 0:
            return Answer(text="No documents have been indexed yet.", confidence="Low")

        # 1. Check the saved/approved Q&A cache first — instant, no API call.
        if use_cache:
            cached = storage.find_saved_qa(question)
            if cached:
                answer = Answer(text=cached["answer"], confidence="High", from_cache=True)
                storage.log_qa(user, ",".join(allowed_roles), question, answer.text, "High (cached)", [])
                return answer

        # 2. Retrieve relevant chunks (filtered to what this role can see).
        sources = self._retrieve(question, allowed_roles, top_k)
        confidence = self._confidence_from(sources)

        context = "\n\n".join(
            f"[{s.doc_name}, Page {s.page}] {s.text}" for s in sources
        ) or "(no relevant passages found)"

        history_block = ""
        if chat_history:
            recent = chat_history[-4:]
            formatted = "\n".join(f"{role}: {msg}" for role, msg in recent)
            history_block = f"\nRecent conversation for context (use only to resolve pronouns/follow-ups, not as a source of facts):\n{formatted}\n"

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=MODEL,
            contents=ANSWER_PROMPT.format(
                history_block=history_block, context=context, question=question
            ),
        )
        text_out = response.text

        citations = [{"doc": s.doc_name, "page": s.page} for s in sources]
        storage.log_qa(user, ",".join(allowed_roles), question, text_out, confidence, citations)

        return Answer(text=text_out, confidence=confidence, sources=sources)

    # ---------------- version comparison ----------------

    def compare_versions(self, question: str, doc_id_a: str, doc_id_b: str, api_key: str,
                          label_a: str = "A", label_b: str = "B", top_k: int = 4) -> str:
        sources_a = self._retrieve(question, allowed_roles=[], top_k=top_k, doc_id_filter=doc_id_a)
        sources_b = self._retrieve(question, allowed_roles=[], top_k=top_k, doc_id_filter=doc_id_b)

        context_a = "\n\n".join(f"[Page {s.page}] {s.text}" for s in sources_a) or "(no relevant passages found)"
        context_b = "\n\n".join(f"[Page {s.page}] {s.text}" for s in sources_b) or "(no relevant passages found)"

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=MODEL,
            contents=COMPARE_PROMPT.format(
                question=question, label_a=label_a, label_b=label_b,
                context_a=context_a, context_b=context_b,
            ),
        )
        return response.text


# ---------------- PDF chunking helpers ----------------

@dataclass
class _RawChunk:
    text: str
    page: int


def _extract_chunks(file_bytes: bytes) -> list[_RawChunk]:
    chunks: list[_RawChunk] = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            for para in _split_into_paragraphs(text):
                if para.strip():
                    chunks.append(_RawChunk(text=para.strip(), page=page_num))
    return chunks


def _split_into_paragraphs(text: str, target_len: int = 700) -> list[str]:
    raw_paragraphs = re.split(r"\n\s*\n", text)
    pieces: list[str] = []
    buffer = ""

    for para in raw_paragraphs:
        para = para.replace("\n", " ").strip()
        if not para:
            continue
        if len(buffer) + len(para) <= target_len:
            buffer = f"{buffer} {para}".strip()
        else:
            if buffer:
                pieces.append(buffer)
            if len(para) > target_len:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                sub = ""
                for s in sentences:
                    if len(sub) + len(s) <= target_len:
                        sub = f"{sub} {s}".strip()
                    else:
                        if sub:
                            pieces.append(sub)
                        sub = s
                buffer = sub
            else:
                buffer = para
    if buffer:
        pieces.append(buffer)
    return pieces
