# Security policy

## Supported versions

graphwalk is pre-1.0. Only the latest release (and the default branch) gets security
fixes.

## Reporting a vulnerability

Please report privately through GitHub's
[private vulnerability reporting](https://github.com/manvirchakal/graphwalk/security/advisories/new).
Do not open a public issue. Include what is affected, how to reproduce it, and the
impact you expect. You should get an acknowledgement within 7 days, and a fix or a
plan within 30 days for confirmed issues. Reporters are credited unless they ask not
to be.

## How graphwalk handles secrets

These are the guarantees we design and test for. A break in any of them is a
vulnerability.

- **Provider API keys** (OpenRouter, TypeSafe, OpenAI, Anthropic, x.ai) come only from
  the environment (library, CLI, stdio MCP) or from request headers (remote MCP).
  They are never written to disk, graph files, caches, results, or logs.
- **Remote MCP**: keys sent in headers are used only for that client's requests and
  held in memory only (in a bounded cache of backends, keyed by a hash). Provider
  error messages are redacted before they reach clients or logs. Server access has
  its own authentication (a bearer token or OAuth 2.1 access tokens), separate from
  provider keys, and the server refuses to listen beyond localhost without it. Base
  URLs sent in headers are honoured only if on the operator's allowlist, which
  prevents server-side request forgery. Server-side keys are used only if the
  operator enables that. Remote `ingest` reads server paths only under an
  operator-chosen root. The container runs as a non-root user.
- **Test suite**: the default `pytest` run blocks all network sockets, so tests
  cannot leak keys to real services.

## Out of scope

- Content a model or decision backend returns (answers, extracted facts). graphwalk
  records provenance so you can check it, but does not vouch for it.
- Denial of service against your own self-hosted server through expensive queries;
  use the request limits in the server configuration.
- Vulnerabilities in the model providers themselves.
