# selfMCP

A self-extending MCP server — a skill registry, search index, and execution
runtime that any connected LLM client (Claude Desktop, Claude.ai, Cursor, …)
can grow over time by calling a small set of bootstrap tools.

The core idea: ship eight fixed tools. Let the model build everything else.

## How it works

1. The server starts with a small, fixed toolset (the "bootstrap" tools).
2. A connected LLM calls `skill_list_summary` to see what's already in the
   registry — a compact `[{id, name, short_description}]` index.
3. When it needs something specific, it calls `skill_search(query)` to find
   relevant skills (FTS5 keyword + embedding vector similarity, merged).
4. For any promising hit it calls `skill_get_detail(skill_id)` to load the
   full body on demand — keeps context lean.
5. If nothing matches, the LLM writes a new skill (a SKILL.md with a Python
   code block) and stores it via `skill_create`. Next session it's there.
6. Skills can be executed in a sandboxed subprocess via `skill_execute`, or
   the client can just read the body and run it locally.

Everything lives in a single SQLite file — no external services required.

## Bootstrap tools

### CRUD layer

| Tool            | Purpose |
|-----------------|---------|
| `skill_create`  | Insert a new skill (name, description, body, deps, auth_config). Seeds FTS, embeddings, and v1 in `skill_versions`. |
| `skill_update`  | Patch an existing skill by id. Archives the previous row and bumps `version`. |
| `skill_delete`  | Soft-delete. Row stays in `skills` with `is_active=0`; history preserved in `skill_versions`. |
| `skill_execute` | Run a skill body as a Python subprocess with `SELFMCP_PARAMS` env. Returns `{stdout, stderr, exit_code, timed_out}`. |

### Discovery layer

| Tool                 | Purpose |
|----------------------|---------|
| `skill_list_summary` | Materialized table of contents: `[{id, name, short_description}]`. Cheap to inject into context. |
| `skill_get_detail`   | Full record for a single skill: body, deps, auth_config, version, timestamps. |
| `skill_search`       | Hybrid FTS5 + vector search. Modes: `keyword`, `vector`, `hybrid` (default). |
| `skill_auth_url`     | For skills with `auth_config`, returns either an API-key instruction string or a prefilled OAuth2 URL. |

## Storage

SQLite with FTS5 — single file, zero ops. Schema:

```
skills(id, name, description, body, dependencies_json,
       auth_config_json, version, is_active, created_at, updated_at)

skill_versions(id, skill_id, version, name, description, body,
               dependencies_json, auth_config_json, changed_at)

skills_fts(name, description)                -- FTS5 virtual table
skill_embeddings(skill_id, vector, dim, model)  -- float32 blob
skill_credentials(skill_id, key, value, created_at)
skill_summary_cache(id, summary_json, updated_at)
```

Embeddings are stored as raw float32 blobs. Cosine similarity is computed in
Python at query time. At a few hundred skills this is fast enough; at scale
swap in a vector index.

## Embeddings

`embeddings.py` tries LiteLLM first (defaulting to OpenAI's
`text-embedding-3-small`). If no API key is configured it falls back to a
deterministic hash-based 256-dim embedding so the server still runs offline
and tests are reproducible.

Enable real embeddings by exporting:

```bash
export OPENAI_API_KEY=sk-...
# optional override
export SELFMCP_EMBED_MODEL=text-embedding-3-small
```

## Auth for external skills

Each skill can declare an `auth_config`. Two types are built in:

```json
{
  "type": "api_key",
  "env_var": "OPENAI_API_KEY",
  "instructions": "Get a key at https://platform.openai.com/api-keys"
}
```

```json
{
  "type": "oauth2",
  "auth_url":  "https://accounts.google.com/o/oauth2/v2/auth",
  "token_url": "https://oauth2.googleapis.com/token",
  "scopes":    ["https://www.googleapis.com/auth/calendar.readonly"],
  "client_id_env":     "GOOGLE_CLIENT_ID",
  "client_secret_env": "GOOGLE_CLIENT_SECRET"
}
```

When `skill_execute` is called, the server checks whether the required env
vars are set. If not, it returns a `missing_credentials` error pointing at
`skill_auth_url`, which returns human-readable instructions (or, for OAuth2,
a pre-filled authorization URL). The client renders that as a clickable
link for the user.

## Running locally

```bash
pip install -r requirements.txt
python3 server.py
```

By default the server binds `0.0.0.0:8000` with the streamable-HTTP transport
mounted at `/mcp`. Change the transport with `SELFMCP_TRANSPORT`:

```bash
SELFMCP_TRANSPORT=stdio          python3 server.py   # for stdio clients
SELFMCP_TRANSPORT=sse            python3 server.py   # legacy SSE transport
SELFMCP_TRANSPORT=streamable-http python3 server.py  # default
```

All env vars:

| Var                 | Default                 | Purpose                               |
|---------------------|-------------------------|---------------------------------------|
| `SELFMCP_DB_PATH`   | `./selfmcp.db`          | SQLite file location                  |
| `SELFMCP_TRANSPORT` | `streamable-http`       | `stdio`, `sse`, or `streamable-http`  |
| `SELFMCP_HOST`      | `0.0.0.0`               | HTTP bind host                        |
| `SELFMCP_PORT`      | `8000`                  | HTTP bind port (falls back to `PORT`) |
| `SELFMCP_EMBED_MODEL` | `text-embedding-3-small` | LiteLLM embedding model            |
| `SELFMCP_USE_LITELLM` | _(unset)_             | Force LiteLLM without an OPENAI key   |
| `OPENAI_API_KEY`    | _(unset)_               | Enables remote LiteLLM embeddings     |

## Running on Replit

1. **Fork / import** the repo into Replit (use "Import from GitHub").

2. Replit auto-detects `requirements.txt` and installs dependencies. If it
   doesn't, open the Shell and run:

   ```bash
   pip install -r requirements.txt
   ```

3. The included `.replit` file sets `SELFMCP_TRANSPORT=streamable-http` and
   uses `python3 server.py` as the run command. Just press **Run**.

4. The server picks up Replit's injected `PORT` environment variable
   automatically, so no manual port configuration is needed.

5. **(Optional)** Add secrets in **Tools → Secrets**:

   | Secret key          | Value                        |
   |---------------------|------------------------------|
   | `OPENAI_API_KEY`    | `sk-...` (enables real embeddings) |
   | `SELFMCP_DB_PATH`   | `/home/user/selfmcp.db` (explicit persistent path) |

6. **Connect Claude.ai** — once the Repl is running, copy the public URL
   (shown in the Webview tab, e.g. `https://<repl-name>.<username>.repl.co`)
   and register it as a remote MCP server in Claude.ai:

   ```
   https://<repl-name>.<username>.repl.co/mcp
   ```

   In Claude Desktop you can also point at the Replit URL instead of running
   the server locally:

   ```json
   {
     "mcpServers": {
       "selfmcp": {
         "url": "https://<repl-name>.<username>.repl.co/mcp"
       }
     }
   }
   ```

> **Persistence note:** Replit's filesystem is persistent across runs for
> private Repls. The SQLite database (`selfmcp.db`) will survive restarts.
> If you're using a free Repl that may be reset, set `SELFMCP_DB_PATH` to a
> path inside a mounted volume, or export/import the `.db` file periodically.

## Connecting from Claude

Add the server to Claude Desktop (`~/.config/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "selfmcp": {
      "command": "python3",
      "args": ["/absolute/path/to/selfMCP/server.py"],
      "env": { "SELFMCP_TRANSPORT": "stdio" }
    }
  }
}
```

Or register as a remote MCP server in Claude.ai pointing at
`http://your-host:8000/mcp`.

## Tests

```bash
pip install pytest
pytest -q
```

`test_selfmcp.py` exercises the storage, versioning, search, and execution
layers against an isolated temp database.

## Project layout

```
server.py       FastMCP entry point — registers the 8 bootstrap tools
skills.py       Plain-Python core logic (what the decorators wrap)
db.py           SQLite schema + connection helper
embeddings.py   LiteLLM embeddings with offline hash fallback
executor.py     Subprocess sandbox for skill_execute
test_selfmcp.py Tests for the core logic
```

## Design notes

- **Two-phase retrieval.** `skill_list_summary` gives a cheap overview;
  `skill_search` narrows it down; `skill_get_detail` loads a full body only
  when needed. At 200+ skills this keeps context budgets sane.
- **Versioning is cheap.** Every `skill_update` and `skill_delete` writes
  the prior row to `skill_versions`, so the LLM can diff, explain, or roll
  back history on demand.
- **Why SQLite + FTS5 rather than a vector DB?** Single file, no ops, deploys
  to Replit trivially. When you outgrow linear cosine sim, swap the
  `_embedding_upsert` / search implementation for a proper index — the
  rest of the code doesn't care.
- **Why SSE / streamable-HTTP rather than stdio by default?** The server
  persists state (the skill database). You want it as a long-lived process
  that multiple clients can share.
