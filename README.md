# Kortex

A personal research assistant that lets you chat with your own notes and files. Ask questions in plain English — it searches your local database first, then falls back to reading files from your vault.

Built with LangGraph, Gemini 2.5 Flash, and FastMCP. Comes with both a terminal interface and a Streamlit UI.

## How it works

Kortex runs a local MCP server (`server.py`) that exposes two tools to the agent:

- **search_notes** — full-text search over a SQLite database of research entries
- **read_file** — reads `.txt` files from a `vault/` folder

When you ask a question, the agent decides which tool to call (or both), then gives you a grounded answer. The Streamlit UI shows the full thought process — which tools were called, what inputs were passed, and what came back.

## Setup

**1. Clone and install dependencies**

```bash
git clone <repo>
cd kortex
uv sync
```

**2. Add your Gemini API key**

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_key_here
```

Get a key at [aistudio.google.com](https://aistudio.google.com).

**3. Seed some data**

```bash
# Create the database and add a note
uv run python -c "
import sqlite3
conn = sqlite3.connect('kortex.db')
conn.execute('CREATE TABLE IF NOT EXISTS notes (id INTEGER PRIMARY KEY, title TEXT, content TEXT, created_at TEXT)')
conn.execute(\"INSERT OR IGNORE INTO notes VALUES (1, 'My First Note', 'Some research content here.', '2026-04-25')\")
conn.commit()
"

# Add a file to the vault
mkdir -p vault
echo "Your notes go here." > vault/notes.txt
```

## Running

**Terminal**
```bash
uv run python agent.py
```

**Streamlit UI**
```bash
uv run streamlit run main.py
```

Opens at `http://localhost:8501`. The UI shows a "Thought process" expander under each response so you can see exactly which tools the agent called.

## Project structure

```
kortex/
├── server.py      # FastMCP server — defines the search_notes and read_file tools
├── agent.py       # LangGraph ReAct agent + terminal chat loop
├── main.py        # Streamlit UI
├── kortex.db      # SQLite database (created by you)
└── vault/         # Folder for .txt files the agent can read
```

## Database schema

The `notes` table expects these columns:

```sql
CREATE TABLE notes (
    id         INTEGER PRIMARY KEY,
    title      TEXT,
    content    TEXT,
    created_at TEXT
);
```
