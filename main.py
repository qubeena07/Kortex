import asyncio
import json
import threading

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from agent import SYSTEM_PROMPT

# ── page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Kortex",
    page_icon="⬡",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── dark theme CSS ────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* palette */
    :root {
        --bg0:     #0d0d11;
        --bg1:     #13131a;
        --bg2:     #1c1c27;
        --bg3:     #252535;
        --accent:  #7c6af7;
        --accent2: #a78bfa;
        --border:  #2a2a3d;
        --text:    #c9d1f0;
        --muted:   #5a5a80;
    }

    /* app background */
    .stApp, [data-testid="stAppViewContainer"] {
        background-color: var(--bg0) !important;
    }
    .main .block-container {
        max-width: 780px;
        padding-top: 0;
    }

    /* hide default streamlit header chrome */
    [data-testid="stHeader"] { background: transparent !important; }
    #MainMenu, footer { visibility: hidden; }

    /* header */
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

    /* chat messages */
    [data-testid="stChatMessage"] {
        background: var(--bg2) !important;
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.6rem;
    }

    /* thought-process expander */
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

    /* chat input */
    [data-testid="stChatInput"] {
        background: var(--bg2) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px;
    }
    [data-testid="stChatInput"] textarea {
        background: transparent !important;
        color: var(--text) !important;
    }

    /* tool call cards inside expander */
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
    </style>
    """,
    unsafe_allow_html=True,
)

# ── header ────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="kortex-header">
        <h1>⬡ KORTEX</h1>
        <p>Personal Research Assistant</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── agent init (cached across reruns) ─────────────────────────────────────────


@st.cache_resource(show_spinner="Starting Kortex tools...")
def load_agent():
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()

    async def _init():
        client = MultiServerMCPClient(
            {
                "kortex": {
                    "command": "uv",
                    "args": ["run", "python", "server.py"],
                    "transport": "stdio",
                }
            }
        )
        tools = await client.get_tools()
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        return create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)

    agent = asyncio.run_coroutine_threadsafe(_init(), loop).result(timeout=30)
    return loop, agent


# ── agent query ───────────────────────────────────────────────────────────────


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            b["text"] for b in content if isinstance(b, dict) and b.get("type") == "text"
        )
    return str(content)


def query_agent(loop, agent, lc_messages):
    """
    Run agent on lc_messages. Returns (response_text, tool_steps).
    tool_steps: list of {"tool": str, "input": dict, "output": str|None}
    """
    input_len = len(lc_messages)

    async def _run():
        return await agent.ainvoke({"messages": lc_messages})

    result = asyncio.run_coroutine_threadsafe(_run(), loop).result(timeout=120)
    new_msgs = result["messages"][input_len:]

    tool_steps = []
    final_response = ""

    for msg in new_msgs:
        kind = msg.__class__.__name__
        if kind == "AIMessage":
            if getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    tool_steps.append({"tool": tc["name"], "input": tc["args"], "output": None})
            else:
                final_response = _extract_text(msg.content)
        elif kind == "ToolMessage":
            for step in reversed(tool_steps):
                if step["output"] is None:
                    step["output"] = _extract_text(msg.content)
                    break

    return final_response, tool_steps


# ── render helpers ────────────────────────────────────────────────────────────


def render_tool_steps(steps):
    for i, step in enumerate(steps, 1):
        tool_input = json.dumps(step["input"], indent=2) if step["input"] else "{}"
        output = step["output"] or "—"
        if len(output) > 600:
            output = output[:600] + "\n… (truncated)"

        st.markdown(
            f"""
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
            """,
            unsafe_allow_html=True,
        )


# ── session state ─────────────────────────────────────────────────────────────

if "display_msgs" not in st.session_state:
    st.session_state.display_msgs = []
    # list of {"role": "user"|"assistant", "content": str, "steps": list}

if "lc_msgs" not in st.session_state:
    st.session_state.lc_msgs = []
    # list of {"role": "user"|"assistant", "content": str} — passed to agent

# ── load agent ────────────────────────────────────────────────────────────────

try:
    loop, agent = load_agent()
except Exception as e:
    st.error(f"Agent failed to start: {e}")
    st.info("Check that `GOOGLE_API_KEY` is set and `server.py` is in the project root.")
    st.stop()

# ── render chat history ───────────────────────────────────────────────────────

for msg in st.session_state.display_msgs:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("steps"):
            n = len(msg["steps"])
            label = f"🔍 Thought process — {n} tool call{'s' if n != 1 else ''}"
            with st.expander(label):
                render_tool_steps(msg["steps"])

# ── handle new input ──────────────────────────────────────────────────────────

if prompt := st.chat_input("Ask Kortex anything..."):
    # user bubble
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.display_msgs.append({"role": "user", "content": prompt, "steps": []})
    st.session_state.lc_msgs.append({"role": "user", "content": prompt})

    # assistant bubble with spinner while agent runs
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response, steps = query_agent(loop, agent, st.session_state.lc_msgs)
            except Exception as e:
                response = f"⚠️ Error: {e}"
                steps = []

        st.markdown(response)

        if steps:
            n = len(steps)
            label = f"🔍 Thought process — {n} tool call{'s' if n != 1 else ''}"
            with st.expander(label):
                render_tool_steps(steps)

    st.session_state.lc_msgs.append({"role": "assistant", "content": response})
    st.session_state.display_msgs.append({"role": "assistant", "content": response, "steps": steps})
