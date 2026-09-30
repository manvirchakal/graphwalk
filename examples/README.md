# Using graphwalk from an agent (MCP)

graphwalk's MCP server gives any MCP client these tools: `locate`, `read`, `neighbors`,
`get_node`, `ingest`, `ingest_status`, and `status`. The usual loop is: `locate` the
question, `read` the best locations, answer from that text.

Install with the extras: `pip install 'graphwalk[mcp,llm]'` (add `embeddings` for
`dense`/`hybrid` locate with a local model).

## stdio (a local graph file)

Keys come from the server's environment. Generic client config:
[`mcp/stdio.json`](mcp/stdio.json). In Claude Code:

```bash
claude mcp add --env OPENROUTER_API_KEY=sk-or-... --transport stdio graphwalk \
  -- graphwalk mcp --db ~/graphs/my.db
```

## Remote HTTP (a shared server or the container)

Start a server. It refuses to bind beyond localhost without auth:

```bash
export GRAPHWALK_AUTH=bearer GRAPHWALK_AUTH_TOKEN=$(openssl rand -hex 24)
graphwalk mcp --http --db my.db --host 0.0.0.0 --port 8080
# or: docker run -p 8080:8080 -v graphwalk-data:/data \
#       -e GRAPHWALK_AUTH=bearer -e GRAPHWALK_AUTH_TOKEN=... ghcr.io/<owner>/graphwalk
```

Clients send their own provider keys as headers (`X-OpenRouter-API-Key`,
`X-Anthropic-API-Key`, ...; any setting from [`.env.example`](../.env.example) works,
e.g. `X-Graphwalk-LLM-Provider`). The server uses them for that client's requests only,
keeps them in memory, and never logs them. Its own keys are used only if the operator
sets `GRAPHWALK_SERVER_KEYS=true`. Generic config: [`mcp/http.json`](mcp/http.json). In
Claude Code:

```bash
claude mcp add --transport http graphwalk https://graphwalk.example.com/mcp \
  --header "Authorization: Bearer $GRAPHWALK_AUTH_TOKEN" \
  --header "X-OpenRouter-API-Key: $OPENROUTER_API_KEY"
```

Use hyphenated header names: proxies such as nginx drop headers with underscores.

### Server settings

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

[`harness.py`](harness.py) is a small tool-calling agent (about 150 lines) that talks to
graphwalk over either transport. It indexes the [`data/curies`](data/curies) folder, then
answers:

```bash
python examples/harness.py --stdio --db curies.db --ingest examples/data/curies \
  "Where was Marie Curie's husband born?"
# [ingest] 3 documents, 3 new, 16 entities, cost $0.0021
# [tool] locate {"query": "...", ...}
# [tool] read {"key": "curies/pierre-curie.md", "start": 53, "end": 82, ...}
# Marie Curie's husband, Pierre Curie, was born in Paris. [curies/pierre-curie.md]

python examples/harness.py --http http://127.0.0.1:8080/mcp --token "$GRAPHWALK_AUTH_TOKEN" \
  --ingest examples/data/curies "Who discovered piezoelectricity, and in what year?"
```

Over HTTP the folder is sent as inline documents (a remote server cannot read your
disk) and the key travels as a header.
