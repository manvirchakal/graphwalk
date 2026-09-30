"""A minimal tool-calling agent that uses graphwalk over MCP, written from scratch.

It connects to graphwalk (stdio subprocess or remote HTTP), offers graphwalk's tools to
a chat model, and loops until the model answers. Keys come only from the environment:
over stdio they are inherited by the server process; over HTTP they are sent as
headers. Nothing is written to disk.

    # stdio: index a folder, then ask
    python examples/harness.py --stdio --db curies.db --ingest examples/data/curies \\
        "Where was Marie Curie's husband born?"

    # remote: the same against a running server (`graphwalk mcp --http`)
    python examples/harness.py --http http://127.0.0.1:8080/mcp --token "$TOKEN" \\
        --ingest examples/data/curies "Who won the 1935 Nobel Prize in Chemistry?"

Needs the ``mcp`` and ``llm`` extras and ``OPENROUTER_API_KEY`` (used for graphwalk's
models and for the agent's own chat model, ``--model``).
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx2
from mcp import Client, StdioServerParameters
from mcp.client.streamable_http import streamable_http_client

from graphwalk.llm.litellm_import import import_litellm

SYSTEM = (
    "Answer the user's question from the indexed documents. Use `locate` to find where "
    "the answer is, then `read` the promising locations (pass key, start, end, doc_hash "
    "exactly). Answer in one or two sentences and cite the document keys you used."
)
MAX_STEPS = 8


def tool_specs(tools: Any) -> list[dict[str, Any]]:
    """MCP tool definitions as OpenAI-style function tools."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.input_schema,
            },
        }
        for t in tools.tools
    ]


def result_text(result: Any) -> str:
    if result.structured_content is not None:
        return json.dumps(result.structured_content)
    return "\n".join(getattr(c, "text", "") for c in result.content)


async def ingest(client: Client, folder: Path, *, inline: bool) -> None:
    """Index ``folder``: by path over stdio; as inline documents over HTTP (a remote
    server cannot read your disk)."""
    if inline:
        documents = [
            {"id": p.name, "text": p.read_text(encoding="utf-8"), "title": p.stem}
            for p in sorted(folder.iterdir())
            if p.suffix in (".md", ".txt")
        ]
        arguments: dict[str, Any] = {"documents": documents, "source_id": folder.name}
    else:
        arguments = {"path": str(folder.resolve()), "source_id": folder.name}
    job = json.loads(result_text(await client.call_tool("ingest", arguments)))
    while True:
        status = json.loads(
            result_text(await client.call_tool("ingest_status", {"job_id": job["job_id"]}))
        )
        if status["state"] != "running":
            break
        await asyncio.sleep(1)
    if status["state"] != "done":
        sys.exit(f"ingest failed: {status['error']}")
    report = status["report"]
    print(
        f"[ingest] {report['documents']} documents, {report['new']} new, "
        f"{report['nodes_created']} entities, cost ${report['cost_usd'] or 0:.4f}"
    )


async def ask(client: Client, question: str, model: str) -> str:
    litellm = import_litellm()
    tools = tool_specs(await client.list_tools())
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]
    cost = 0.0
    for _ in range(MAX_STEPS):
        response = await litellm.acompletion(
            model=model,
            messages=messages,
            tools=tools,
            api_key=os.environ["OPENROUTER_API_KEY"],
            usage={"include": True},
        )
        cost += float(getattr(response.usage, "cost", 0) or 0)
        message = response.choices[0].message
        messages.append(message.model_dump(exclude_none=True))
        if not message.tool_calls:
            print(f"[agent] cost ${cost:.4f}")
            return str(message.content)
        for call in message.tool_calls:
            arguments = json.loads(call.function.arguments or "{}")
            print(f"[tool] {call.function.name} {json.dumps(arguments)[:120]}")
            result = await client.call_tool(call.function.name, arguments)
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": result_text(result)[:8000]}
            )
    return "(no answer within the step limit)"


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--stdio", action="store_true", help="launch `graphwalk mcp` locally")
    mode.add_argument("--http", metavar="URL", help="a remote server's MCP endpoint")
    parser.add_argument("--db", default="graphwalk.db", help="stdio: the SQLite graph")
    parser.add_argument("--token", help="http: the server's bearer token")
    parser.add_argument("--ingest", type=Path, help="index this folder first")
    parser.add_argument("--model", default="openrouter/openai/gpt-6-luna")
    parser.add_argument("question")
    args = parser.parse_args()
    if "OPENROUTER_API_KEY" not in os.environ:
        sys.exit("set OPENROUTER_API_KEY")

    if args.stdio:
        server: Any = StdioServerParameters(
            command="graphwalk", args=["mcp", "--db", args.db], env=dict(os.environ)
        )
    else:
        headers = {"X-OpenRouter-API-Key": os.environ["OPENROUTER_API_KEY"]}
        if args.token:
            headers["Authorization"] = f"Bearer {args.token}"
        http = httpx2.AsyncClient(headers=headers, timeout=120)
        server = streamable_http_client(args.http, http_client=http)
    async with Client(server) as client:
        if args.ingest:
            await ingest(client, args.ingest, inline=not args.stdio)
        print(await ask(client, args.question, args.model))


if __name__ == "__main__":
    asyncio.run(main())
