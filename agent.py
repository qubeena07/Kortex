import asyncio
import sys
import logging

from dotenv import load_dotenv
load_dotenv()

logging.getLogger("fastmcp").setLevel(logging.WARNING)
logging.getLogger("mcp").setLevel(logging.WARNING)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langgraph")

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

SYSTEM_PROMPT = """You are Kortex, a personal research assistant with access to two tools:

- search_notes: searches a local SQLite database of research entries
- read_file: reads a .txt file from the local vault folder

When answering questions:
1. ALWAYS call search_notes first.
2. If search_notes returns "No notes found", you MUST then call read_file with "about.txt" to check the vault before giving up.
3. Only tell the user you have no information after trying both tools.

Be concise. Cite the note ID or filename when referencing retrieved content."""


async def run_agent():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    client = MultiServerMCPClient({
        "kortex": {
            "command": "uv",
            "args": ["run", "python", "server.py"],
            "transport": "stdio",
            "env": {"FASTMCP_LOG_LEVEL": "WARNING"},
        }
    })
    tools = await client.get_tools()
    agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)

    print("Kortex ready. Type 'exit' to quit.\n")
    history = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            print("Bye.")
            break

        history.append({"role": "user", "content": user_input})

        try:
            result = await agent.ainvoke({"messages": history})
            content = result["messages"][-1].content
            response = (
                " ".join(b["text"] for b in content if isinstance(b, dict) and b.get("type") == "text")
                if isinstance(content, list) else content
            )
            history.append({"role": "assistant", "content": response})
            print(f"\nKortex: {response}\n")
        except Exception as e:
            print(f"\nError: {e}\n", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(run_agent())
