import streamlit as st
import os
from pathlib import Path
from groq import Groq
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

load_dotenv()

# ── Paths
BASE_DIR   = Path(__file__).resolve().parent.parent
VECTOR_DIR = BASE_DIR / "vectorstore"

# ── Page config 
st.set_page_config(
    page_title = "VidIQ-RAG | AI Video Consultant",
    page_icon  = "🎬",
    layout     = "wide"
)

# ── CSS 
st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; }
    header { visibility: hidden; }

    .header-card {
        background: linear-gradient(135deg, #1e3a5f, #2d6a9f);
        border-radius: 12px;
        padding: 24px 32px;
        color: white;
        margin-bottom: 24px;
    }
    .header-card h1 { margin: 0; font-size: 28px; }
    .header-card p  { margin: 4px 0 0; opacity: 0.85; font-size: 15px; }

    .answer-card {
        background: white;
        border-left: 5px solid #2d6a9f;
        border-radius: 8px;
        padding: 20px 24px;
        margin: 12px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        color: #1a1a1a;
        font-size: 15px;
        line-height: 1.7;
    }
    .no-answer-card {
        background: #fff5f5;
        border-left: 5px solid #e53e3e;
        border-radius: 8px;
        padding: 20px 24px;
        margin: 12px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        color: #742a2a;
        font-size: 15px;
    }
    .relevance-badge {
        display: inline-block;
        background: #ebf8ff;
        color: #2b6cb0;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
    }
    section[data-testid="stSidebar"] {
        background-color: #f0f4f8;
    }
    .stTextInput input {
        background-color: white !important;
        color: #1a1a1a !important;
        border: 2px solid #cbd5e0 !important;
        border-radius: 8px !important;
        font-size: 15px !important;
        padding: 10px 14px !important;
    }
    .stTextInput input:focus {
        border-color: #2d6a9f !important;
        box-shadow: 0 0 0 3px rgba(45,106,159,0.15) !important;
    }
    .stButton > button {
        background-color: #2d6a9f;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-size: 15px;
        font-weight: 600;
        width: 100%;
        transition: background 0.2s;
    }
    .stButton > button:hover {
        background-color: #1e3a5f;
    }
    hr { border-color: #e2e8f0; }
</style>
""", unsafe_allow_html=True)


# Load resources (cached)
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_collection():
    client = chromadb.PersistentClient(
        path=str(VECTOR_DIR),
        settings=Settings(anonymized_telemetry=False)
    )
    return client.get_collection("vidiq_rag")


@st.cache_resource
def load_groq():
    return Groq(api_key=os.getenv("GROQ_API_KEY_3"))


#System prompt
SYSTEM_PROMPT = """You are a knowledge assistant built from video content.

STRICT RULES:
1. Answer ONLY from the context provided below.
2. Do NOT use your general knowledge under any circumstances.
3. If the answer is not found in the context, respond with exactly:
   "I don't know based on the provided video content."
4. Every answer MUST cite the video title and timestamp.
5. Be concise and direct.

SEMANTIC MATCHING RULES:
- If the user asks about "penalties" look for consequences, enforcement,
  actions, fees, charges, freezing, garnishing, seizing in the context.
- If the user asks about "rights" look for legal protections, what you
  can do, options available in the context.
- If the user asks about "options" look for choices, alternatives,
  solutions, ways to handle in the context.
- If the user asks about "cost" look for fees, interest, charges,
  amounts, rates in the context.
- If the user asks about "contact" look for who should i turn to, departments, agencies,
  professionals in the context.
- Generally: look for the MEANING of the question, not just exact words.
- If the context contains information that answers the question even with
  different wording, USE IT to answer."""




# ── Retrieve chunks
def retrieve(query: str, collection, embedder, n: int = 4) -> list:
    emb = embedder.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[emb],
        n_results=n
    )

    chunks = []
    if results and results["documents"][0]:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            chunks.append({
                "text"    : doc,
                "metadata": meta,
                "score"   : round(1 - dist, 4)
            })

    return chunks


# ── Generate answer 
def generate_answer(query: str, chunks: list, groq_client) -> str:
    if not chunks:
        return "I don't know based on the provided video content."

    context = "\n\n".join([
        f"--- SOURCE {i+1} ---\n"
        f"Video : {c['metadata']['title']}\n"
        f"Time  : {c['metadata']['timestamp']}\n"
        f"URL   : {c['metadata']['url']}\n"
        f"Content:\n{c['text'][:400]}"
        for i, c in enumerate(chunks)
    ])

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"CONTEXT:\n{context}\n\nQUESTION: {query}"
            }
        ],
        temperature=0.0,
        max_tokens=400
    )
    return response.choices[0].message.content.strip()


# ── Session state init
if "history" not in st.session_state:
    st.session_state.history     = []
if "prefill" not in st.session_state:
    st.session_state.prefill     = ""
if "auto_search" not in st.session_state:
    st.session_state.auto_search = False


# Load resources 
embedder    = load_embedder()
collection  = load_collection()
groq_client = load_groq()


# Header
st.markdown("""
<div class="header-card">
    <h1>🎬 VidIQ-RAG</h1>
    <p>AI Consultant powered by Video Knowledge —
    answers strictly from provided video content</p>
</div>
""", unsafe_allow_html=True)


# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    n_results = st.slider("Sources to retrieve", 1, 10, 3)

    st.divider()
    st.markdown("### 📊 Knowledge Base")
    st.metric("Chunks indexed", collection.count())

    st.divider()
    st.markdown("### 💡 Sample Questions")

    samples = [
        "What are the consequences of unpaid payroll deductions?",
        "Can I negotiate a payment plan with the CRA?",
        "What happens if I ignore CRA payroll debt?",
        "How does payroll debt affect my financing?",
        "What are my legal rights with the CRA?",
        "Who should I contact for CRA payroll problems?",
        "How can I walk away from my payroll debt?",
        "What penalties does the CRA apply for unpaid deductions?",
        "How do I file a personal income tax return in australia?",
        "What is the corporate tax rate in the United States?"
    ]

    for q in samples:
        if st.button(q, key=f"sample_{q}", use_container_width=True):
            st.session_state.prefill     = q
            st.session_state.auto_search = True
            st.rerun()

    st.divider()
    if st.button("🗑️ Clear history", use_container_width=True):
        st.session_state.history     = []
        st.session_state.prefill     = ""
        st.session_state.auto_search = False
        st.rerun()

    st.divider()
    st.markdown("""
    ### ℹ️ About
    Answers strictly from provided
    video content only.

    **Stack:**
    - Groq Whisper — transcription
    - Groq Vision — OCR + visual
    - sentence-transformers — embeddings
    - ChromaDB — vector store
    - Groq LLaMA — answers
    """)


#Search area
col1, col2 = st.columns([5, 1])

with col1:
    query = st.text_input(
        "Ask a question about the video content:",
        value=st.session_state.prefill,
        placeholder=(
            "e.g. What are the consequences of "
            "unpaid payroll deductions?"
        ),
        label_visibility="visible"
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    search = st.button("🔍 Search", use_container_width=True)

# Auto trigger search when sample button clicked
if st.session_state.auto_search and query.strip():
    search                       = True
    st.session_state.auto_search = False
    st.session_state.prefill     = ""


# Process search
if search and query.strip():
    with st.spinner("Searching knowledge base..."):
        chunks = retrieve(query, collection, embedder, n_results)

    with st.spinner("Generating answer..."):
        answer_text = generate_answer(query, chunks, groq_client)

    # Save to history
    st.session_state.history.insert(0, {
        "query"  : query,
        "answer" : answer_text,
        "sources": [
            {
                "title"     : c["metadata"]["title"],
                "url"       : c["metadata"]["url"],
                "timestamp" : c["metadata"]["timestamp"],
                "score"     : c["score"],
                "transcript": c["metadata"].get("transcript_text", "")[:250],
                "ocr"       : c["metadata"].get("ocr_text", "")[:200],
                "visual"    : c["metadata"].get("visual_description", "")[:200],
            }
            for c in chunks
        ]
    })
    st.session_state.history = st.session_state.history[:4]

elif search and not query.strip():
    st.warning("Please enter a question.")


# Display latest result
if st.session_state.history:
    latest       = st.session_state.history[0]
    is_no_answer = "don't know" in latest["answer"].lower()

    st.markdown("### 💬 Answer")

    if is_no_answer:
        st.markdown(f"""
        <div class="no-answer-card">
            ❌ &nbsp; {latest['answer']}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="answer-card">
            {latest['answer']}
        </div>
        """, unsafe_allow_html=True)

        # Sources
        if latest["sources"]:
            st.markdown("### 📎 Sources")
            seen = set()
            for src in latest["sources"]:
                key = f"{src['title']}_{src['timestamp']}"
                if key in seen:
                    continue
                seen.add(key)

                with st.expander(
                    f"📹 {src['title']} — ⏱️ {src['timestamp']}",
                    expanded=False
                ):
                    st.markdown(
                        f"<span class='relevance-badge'>"
                        f"Relevance: {int(src['score']*100)}%"
                        f"</span>",
                        unsafe_allow_html=True
                    )
                    st.markdown(f"**🔗 URL:** {src['url']}")
                    st.markdown(
                        f"**⏱️ Timestamp:** `{src['timestamp']}`"
                    )
                    st.divider()

                    if src.get("transcript"):
                        st.markdown("**📝 Transcript evidence:**")
                        st.markdown(f"> {src['transcript']}")

                    if src.get("ocr"):
                        st.markdown("**🖥️ On-screen text:**")
                        st.code(src["ocr"])

                    if src.get("visual"):
                        st.markdown("**👁️ Visual context:**")
                        st.info(src["visual"])

    #Recent questions 
    if len(st.session_state.history) > 1:
        st.markdown("### 🕘 Recent Questions")
        for item in st.session_state.history[1:]:
            is_old_no = "don't know" in item["answer"].lower()
            icon      = "❌" if is_old_no else "✅"

            with st.expander(
                f"{icon}  {item['query']}", expanded=False
            ):
                st.markdown(f"""
                <div style="background:#f8f9fa;border-radius:6px;
                            padding:12px;color:#333;font-size:14px;
                            line-height:1.6;">
                    {item['answer']}
                </div>
                """, unsafe_allow_html=True)

                if item["sources"] and not is_old_no:
                    st.markdown("**Sources:**")
                    seen = set()
                    for src in item["sources"]:
                        key = f"{src['title']}_{src['timestamp']}"
                        if key in seen:
                            continue
                        seen.add(key)
                        st.markdown(
                            f"- 📹 **{src['title']}** — "
                            f"`{src['timestamp']}` — "
                            f"[Link]({src['url']})"
                        )


# Footer
st.divider()
st.markdown(
    "<p style='text-align:center;color:#999;font-size:12px;'>"
    "VidIQ-RAG — Answers strictly from provided video content only | "
    "Built with Groq + ChromaDB + sentence-transformers"
    "</p>",
    unsafe_allow_html=True
)