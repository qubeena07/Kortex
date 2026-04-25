import sqlite3
from pathlib import Path

from fastmcp import FastMCP

mcp = FastMCP("Kortex-Tools")

DB_PATH = Path("kortex.db")
VAULT_PATH = Path("vault")


@mcp.tool()
def search_notes(query: str) -> str:
    if not DB_PATH.exists():
        return f"Database not found at '{DB_PATH}'. Create 'kortex.db' with a 'notes' table first."

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id, title, content, created_at
            FROM notes
            WHERE title LIKE ? OR content LIKE ?
            ORDER BY created_at DESC
            """,
            (f"%{query}%", f"%{query}%"),
        )

        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return f"No notes found matching '{query}'."

        results = []
        for row in rows:
            results.append(
                f"[ID: {row['id']}] {row['title']}\n"
                f"Created: {row['created_at']}\n"
                f"{row['content']}\n"
                f"{'-' * 40}"
            )

        return f"Found {len(rows)} note(s) for '{query}':\n\n" + "\n".join(results)

    except sqlite3.OperationalError as e:
        return f"Database error: {e}. Ensure 'notes' table exists with columns: id, title, content, created_at."


@mcp.tool()
def read_file(filename: str) -> str:
    if not filename.endswith(".txt"):
        return f"Only .txt files are supported. Got: '{filename}'"

    safe_name = Path(filename).name
    if safe_name != filename:
        return "Invalid filename — path traversal not allowed."

    file_path = VAULT_PATH / safe_name

    if not VAULT_PATH.exists():
        return f"Vault directory '{VAULT_PATH}' does not exist."

    if not file_path.exists():
        return f"File '{safe_name}' not found in vault."

    try:
        return file_path.read_text(encoding="utf-8")
    except OSError as e:
        return f"Could not read file: {e}"


if __name__ == "__main__":
    mcp.run()
