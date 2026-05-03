import sqlite3
import logging
from pathlib import Path

from fastmcp import FastMCP

logging.getLogger("fastmcp").setLevel(logging.WARNING)

mcp = FastMCP("Kortex-Tools")

DB_PATH = Path("kortex.db")
VAULT_PATH = Path("vault")


@mcp.tool()
def search_notes(query: str) -> str:
    """Search saved research notes in the local database by keyword.
    Use this first whenever the user asks about something they may have saved."""
    if not DB_PATH.exists():
        return "Database not found. Create kortex.db with a notes table first."

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, title, content, created_at FROM notes "
            "WHERE title LIKE ? OR content LIKE ? ORDER BY created_at DESC",
            (f"%{query}%", f"%{query}%"),
        )
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return f"No notes found for '{query}'."

        results = [
            f"[ID: {row['id']}] {row['title']}\n{row['content']}\n{'-' * 40}"
            for row in rows
        ]
        return f"Found {len(rows)} result(s):\n\n" + "\n".join(results)

    except sqlite3.OperationalError as e:
        return f"Database error: {e}"


@mcp.tool()
def read_file(filename: str) -> str:
    """Read a .txt file from the vault folder.
    Use this when the user asks to open or read a specific file, or when search_notes returns nothing."""
    if not filename.endswith(".txt"):
        return f"Only .txt files are supported, got: {filename}"

    safe_name = Path(filename).name
    if safe_name != filename:
        return "Invalid filename."

    if not VAULT_PATH.exists():
        return "Vault folder does not exist."

    file_path = VAULT_PATH / safe_name
    if not file_path.exists():
        return f"{safe_name} not found in vault."

    try:
        return file_path.read_text(encoding="utf-8")
    except OSError as e:
        return f"Could not read file: {e}"


if __name__ == "__main__":
    mcp.run()
