import asyncio
import json
import logging
import sqlite3
import threading
import warnings
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

logging.getLogger("fastmcp").setLevel(logging.WARNING)
logging.getLogger("mcp").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langgraph")

import streamlit as st
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from agent import SYSTEM_PROMPT

DB_PATH = Path("kortex.db")
VAULT_PATH = Path("vault")

st.set_page_config(
    page_title="Kortex",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root {
    --bg0: #0d0d11;
    --bg1: #13131a;
    --bg2: #1c1c27;
    --bg3: #252535;
    --accent: #7c6af7;
    --accent2: #a78bfa;
    --border: #2a2a3d;
    --text: #c9d1f0;
    --muted: #5a5a80;
}

.stApp, [data-testid="stAppViewContainer"] {
    background-color: var(--bg0) !important;
}
.main .block-container {
    max-width: 860px;
    padding-top: 0;
}

[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer { visibility: hidden; }

[data-testid="stSidebar"] {
    background-color: var(--bg1) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

.kortex-header {
    text-align: center;
    padding: 2rem 0 1.25rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}
.kortex-header h1 {
    color: var(--accent2);
    font-size: 1.9rem;
    letter-spacing: 0.15em;
    margin: 0;
    font-weight: 700;
}
.kortex-header p {
    color: var(--muted);
    font-size: 0.8rem;
    margin: 0.3rem 0 0;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

[data-testid="stChatMessage"] {
    background: var(--bg2) !important;
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.6rem;
}

[data-testid="stExpander"] {
    background: var(--bg1) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px;
    margin-top: 0.5rem;
}
[data-testid="stExpander"] summary {
    color: var(--muted) !important;
    font-size: 0.8rem;
}

[data-testid="stChatInput"] {
    background: var(--bg2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: var(--text) !important;
}

.tool-card {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.75rem;
    margin-bottom: 0.5rem;
}
.tool-badge {
    display: inline-block;
    background: var(--accent);
    color: #fff;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    padding: 2px 10px;
    border-radius: 20px;
    margin-bottom: 0.5rem;
}
.note-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.6rem 0.8rem;
    margin-bottom: 0.4rem;
    font-size: 0.8rem;
}
.note-title {
    color: var(--accent2);
    font-weight: 600;
    margin-bottom: 2px;
}
.note-date {
    color: var(--muted);
    font-size: 0.7rem;
}
.note-preview {
    color: var(--text);
    margin-top: 4px;
    opacity: 0.8;
}
</style>
""", unsafe_allow_html=True)


# ── DB helpers ──────────────────────────────────────────────────────────────

def get_notes():
    if not DB_PATH.exists():
        return []
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM notes ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def save_note(title: str, content: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO notes (title, content, created_at) VALUES (?, ?, ?)",
        (title, content, str(date.today())),
    )
    conn.commit()
    conn.close()


def save_vault_file(filename: str, text: str):
    VAULT_PATH.mkdir(exist_ok=True)
    (VAULT_PATH / filename).write_text(text, encoding="utf-8")


# ── Agent ────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Starting Kortex tools...")
def load_agent():
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()

    async def _init():
        client = MultiServerMCPClient({
            "kortex": {
                "command": "uv",
                "args": ["run", "python", "server.py"],
                "transport": "stdio",
                "env": {"FASTMCP_LOG_LEVEL": "WARNING"},
            }
        })
        tools = await client.get_tools()
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        return create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)

    agent = asyncio.run_coroutine_threadsafe(_init(), loop).result(timeout=30)
    return loop, agent


def extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(b["text"] for b in content if isinstance(b, dict) and b.get("type") == "text")
    return str(content)


def run_query(loop, agent, messages):
    input_len = len(messages)

    async def _go():
        return await agent.ainvoke({"messages": messages})

    result = asyncio.run_coroutine_threadsafe(_go(), loop).result(timeout=120)
    new_msgs = result["messages"][input_len:]

    steps = []
    response = ""

    for msg in new_msgs:
        kind = msg.__class__.__name__
        if kind == "AIMessage":
            if getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    steps.append({"tool": tc["name"], "input": tc["args"], "output": None})
            else:
                response = extract_text(msg.content)
        elif kind == "ToolMessage":
            for step in reversed(steps):
                if step["output"] is None:
                    step["output"] = extract_text(msg.content)
                    break

    return response, steps


def render_steps(steps):
    for step in steps:
        tool_input = json.dumps(step["input"], indent=2) if step["input"] else "{}"
        output = step["output"] or "—"
        if len(output) > 600:
            output = output[:600] + "\n… (truncated)"

        st.markdown(f"""
        <div class="tool-card">
            <span class="tool-badge">{step['tool']}</span>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.75rem;margin-top:0.25rem">
                <div>
                    <div style="font-size:0.7rem;color:#5a5a80;margin-bottom:2px">INPUT</div>
                    <pre style="font-size:0.75rem;background:#0d0d11;padding:6px 8px;border-radius:6px;margin:0;overflow:auto">{tool_input}</pre>
                </div>
                <div>
                    <div style="font-size:0.7rem;color:#5a5a80;margin-bottom:2px">OUTPUT</div>
                    <pre style="font-size:0.75rem;background:#0d0d11;padding:6px 8px;border-radius:6px;margin:0;overflow:auto;white-space:pre-wrap">{output}</pre>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⬡ Kortex")
    st.markdown("---")

    # Add note form
    with st.expander("➕ Add Note", expanded=False):
        note_title = st.text_input("Title", key="new_note_title", placeholder="Note title")
        note_content = st.text_area("Content", key="new_note_content", placeholder="Write your research...", height=120)
        if st.button("Save Note", use_container_width=True):
            if note_title.strip() and note_content.strip():
                save_note(note_title.strip(), note_content.strip())
                st.success("Saved!")
                st.rerun()
            else:
                st.warning("Title and content required.")

    st.markdown("---")

    # Upload file to vault
    with st.expander("📁 Upload to Vault", expanded=False):
        uploaded = st.file_uploader("Upload .txt file", type=["txt"], key="vault_upload")
        if uploaded is not None:
            text = uploaded.read().decode("utf-8")
            save_vault_file(uploaded.name, text)
            st.success(f"Saved to vault: {uploaded.name}")

    st.markdown("---")

    # Notes history
    st.markdown("**Notes in DB**")
    notes = get_notes()
    if notes:
        for note in notes:
            preview = note["content"][:80] + "…" if len(note["content"]) > 80 else note["content"]
            st.markdown(f"""
            <div class="note-card">
                <div class="note-title">{note['title']}</div>
                <div class="note-date">{note['created_at']}</div>
                <div class="note-preview">{preview}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown('<span style="color:#5a5a80;font-size:0.8rem">No notes yet.</span>', unsafe_allow_html=True)

    # Vault files
    st.markdown("---")
    st.markdown("**Vault Files**")
    vault_files = sorted(VAULT_PATH.glob("*.txt")) if VAULT_PATH.exists() else []
    if vault_files:
        for f in vault_files:
            st.markdown(f'<span style="color:#a78bfa;font-size:0.8rem">📄 {f.name}</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span style="color:#5a5a80;font-size:0.8rem">No files in vault.</span>', unsafe_allow_html=True)


# ── Main chat ─────────────────────────────────────────────────────────────────

st.markdown("""
<div class="kortex-header">
    <h1>⬡ KORTEX</h1>
    <p>Personal Research Assistant</p>
</div>
""", unsafe_allow_html=True)

if "display_msgs" not in st.session_state:
    st.session_state.display_msgs = []

if "lc_msgs" not in st.session_state:
    st.session_state.lc_msgs = []

try:
    loop, agent = load_agent()
except Exception as e:
    st.error(f"Failed to start agent: {e}")
    st.info("Make sure GOOGLE_API_KEY is set and server.py is in the project root.")
    st.stop()

for msg in st.session_state.display_msgs:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("steps"):
            n = len(msg["steps"])
            with st.expander(f"🔍 Thought process — {n} tool call{'s' if n != 1 else ''}"):
                render_steps(msg["steps"])

if prompt := st.chat_input("Ask Kortex anything..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.display_msgs.append({"role": "user", "content": prompt, "steps": []})
    st.session_state.lc_msgs.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response, steps = run_query(loop, agent, st.session_state.lc_msgs)
            except Exception as e:
                response = f"⚠️ Error: {e}"
                steps = []

        st.markdown(response)

        if steps:
            n = len(steps)
            with st.expander(f"🔍 Thought process — {n} tool call{'s' if n != 1 else ''}"):
                render_steps(steps)

    st.session_state.lc_msgs.append({"role": "assistant", "content": response})
    st.session_state.display_msgs.append({"role": "assistant", "content": response, "steps": steps})
