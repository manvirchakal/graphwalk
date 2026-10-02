# Agents and MCP

graphwalk's MCP server gives any MCP client (Claude Code, Claude Desktop, Cursor, your
own harness) the graph as tools.

## What we measured

On WebQSP (100 questions; 30 with a strong agent), one tool-calling loop, same model
and prompt, only the tools varied (section A8 of the [phase 7 results](results-phase7.md)):

| Agent tools | F1, strong agent | $/question | p50 latency |
|---|---|---|---|
| none (closed book) | 0.57 | 0.011 | 7 s |
| `relations` + `neighbors` | 0.71 | 0.042 | 30 s |
| the same + `walk` | **0.75** | 0.032 | 24 s |
| dense search over the same facts | 0.72 | **0.028** | 40 s |

- `walk` cut the strong agent's cost by 25% (95% CI 7–45%) at equal accuracy. The agent
  still verified confident walks; it explored less and reasoned less.
- With a cheap agent model, `walk` saved 70% of tokens on questions where the first walk
  was confident (≥ 0.9), but the cost overall was a wash: the walk's own calls cost about
  as much as the cheap tokens it saved.
- Text search over the same facts was as accurate. graphwalk's advantage is cost and a
  map of the graph, not accuracy.

## Tools

| Tool | Returns |
|---|---|
| `walk` | Answers reached from the question's entities, each with path and confidence; escalated per the server's threshold |
| `neighbors`, `get_node` | Explore an entity's edges and attributes |
| `status` | What is indexed |
| `locate`, `read` | Documents: source spans for a question, then their text |
| `ingest`, `ingest_status` | Add documents (async job) |

The resource `graphwalk://guide` serves the [agent guide](guide.md). The server's
instructions tell the agent the usual loop: `walk` first; at confidence 0.9 or above,
use the answer; below it, verify with `neighbors` (or the walk is escalated, if the
server sets `GRAPHWALK_ESCALATE_BELOW`). `walk` does not apply constraints: the agent
filters its answers.

## stdio: a local graph file

```bash
pip install 'graphwalk[mcp]'
claude mcp add --env OPENROUTER_API_KEY=sk-or-... --transport stdio graphwalk \
  -- graphwalk mcp --db ~/graphs/kg.db
```

Generic client config ([`examples/mcp/stdio.json`](../examples/mcp/stdio.json)):

```json
{
  "mcpServers": {
    "graphwalk": {
      "command": "graphwalk",
      "args": ["mcp", "--db", "/path/to/kg.db"],
      "env": { "OPENROUTER_API_KEY": "sk-or-..." }
    }
  }
}
```

Keys come from the server's environment.

## Remote HTTP: a shared server or the container

The server refuses to bind beyond localhost without auth:

```bash
export GRAPHWALK_AUTH=bearer GRAPHWALK_AUTH_TOKEN=$(openssl rand -hex 24)
graphwalk mcp --http --db kg.db --host 0.0.0.0 --port 8080
# or the container:
docker run -p 8080:8080 -v graphwalk-data:/data \
  -e GRAPHWALK_AUTH=bearer -e GRAPHWALK_AUTH_TOKEN=... ghcr.io/manvirchakal/graphwalk
```

Clients send their own provider keys as headers (`X-OpenRouter-API-Key`,
`X-Anthropic-API-Key`, ...; any setting works as `X-Graphwalk-...`). The server uses
them for that session only, keeps them in memory, and never logs them. Its own keys are
used only if the operator sets `GRAPHWALK_SERVER_KEYS=true`.

```bash
claude mcp add --transport http graphwalk https://graphwalk.example.com/mcp \
  --header "Authorization: Bearer $GRAPHWALK_AUTH_TOKEN" \
  --header "X-OpenRouter-API-Key: $OPENROUTER_API_KEY"
```

Use hyphenated header names: proxies such as nginx drop headers with underscores.

## Server settings

| Variable | Default | |
|---|---|---|
| `GRAPHWALK_DB` | `graphwalk.db` | The SQLite graph (`/data/graphwalk.db` in the container) |
| `GRAPHWALK_AUTH` | `none` | `none` (localhost only), `bearer`, or `oauth` |
| `GRAPHWALK_AUTH_TOKEN` | | Bearer token, 32+ characters |
| `GRAPHWALK_RESOURCE_URL` | | Public URL of `/mcp` (oauth) |
| `GRAPHWALK_OAUTH_ISSUER` | | Your authorization server (oauth) |
| `GRAPHWALK_OAUTH_AUDIENCE` | resource URL | Expected `aud` |
| `GRAPHWALK_OAUTH_SCOPES` | | Required scopes, comma-separated |
| `GRAPHWALK_OAUTH_JWKS_URL` | discovered | JWT signing keys |
| `GRAPHWALK_OAUTH_INTROSPECTION_URL` | | Validate opaque tokens instead (with `..._CLIENT_ID`/`..._CLIENT_SECRET`) |
| `GRAPHWALK_ALLOWED_HOSTS` | | Host names clients use (DNS-rebinding protection) |
| `GRAPHWALK_ALLOWED_BASE_URLS` | | Base URLs clients may point providers at via headers |
| `GRAPHWALK_SERVER_KEYS` | `false` | Let clients use the server's own provider keys |
| `GRAPHWALK_INGEST_ROOT` | | Directory remote `ingest` may read; otherwise inline documents only |
| `GRAPHWALK_MAX_K`, `_MAX_READ_CHARS`, `_SESSION_CONCURRENCY`, `_MAX_INGEST_CHARS` | 50, 20000, 4, 5000000 | Request limits |
| `GRAPHWALK_LOG_LEVEL` | `WARNING` | |

With `oauth`, graphwalk is a resource server (MCP authorization spec): it serves
`/.well-known/oauth-protected-resource/mcp` pointing clients at your issuer, and checks
each token's signature, issuer, audience, expiry, and scopes. It does not issue tokens.


## A harness from scratch

[`examples/harness.py`](../examples/harness.py) is a small tool-calling agent (about
150 lines) that talks to graphwalk over either transport. The harness used for the
measurements above is `graphwalk.eval.agent` (see
[`scripts/eval/agent_arms.py`](../scripts/eval/agent_arms.py)).
