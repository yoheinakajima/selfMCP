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
| `skill_create`  | Insert a new skill (name, description, body, deps, auth_config). Seeds FTS, embeddings, and v1 in `skill_versions`. If a soft-deleted skill with the same name exists, it is reactivated. |
| `skill_update`  | Patch an existing skill by id. Archives the previous row and bumps `version`. |
| `skill_delete`  | Soft-delete. Row stays in `skills` with `is_active=0`; history preserved in `skill_versions`. |
| `skill_execute` | Run a skill body as a Python subprocess. Passes `SELFMCP_PARAMS` and the full server environment (including API keys) to the subprocess. Returns `{stdout, stderr, exit_code, timed_out}`. |

### Discovery layer

| Tool                 | Purpose |
|----------------------|---------|
| `skill_list_summary` | Materialized table of contents: `[{id, name, short_description}]`. Cheap to inject into context. |
| `skill_get_detail`   | Full record for a single skill: body, deps, auth_config, version, timestamps. |
| `skill_search`       | Hybrid FTS5 + vector search. Modes: `keyword`, `vector`, `hybrid` (default). |
| `skill_auth_url`     | For skills with `auth_config`, returns either an API-key instruction string or a prefilled OAuth2 URL. On Replit, includes a direct link to the Secrets panel. |

## Built-in core skills

The server seeds two **core skills** on first startup. They're ordinary
registry entries — searchable, executable, updatable — with two differences:

1. `skill_delete` refuses to remove them (returns `cannot_delete_core_skill`).
2. `skill_update` refuses to rename them (their name is fixed so the
   core-skill lookup stays stable). Body, description, dependencies, and
   `auth_config` can still be edited.

You can update the body of a core skill to customize it, but it will always
stay in the registry.

### `selfmcp_about`

Structured self-documentation dump covering: source code location, interface
type (MCP-only, no UI), transport options, persistence, versioning/soft-delete
behavior, execution model, auth, all 8 bootstrap tools, search modes, and
Replit setup steps.

```
# Find the skill id (usually 1 on a fresh install):
skill_search("selfmcp_about")

# Run it for the full doc:
skill_execute(skill_id=<id>)

# Or focus on one section:
skill_execute(skill_id=<id>, params={"section": "versioning"})
```

Available sections: `source_code`, `interface`, `transport`, `persistence`,
`versioning`, `execution`, `auth`, `bootstrap_tools`, `search`, `replit_setup`.

### `selfmcp_env_keys`

Reports which API keys / credential environment variables are available to
the server. **Only names are returned — values are never exposed.** The
output includes two lists:

- `known_services` — a curated map of well-known credential env vars
  (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GITHUB_TOKEN`, `SLACK_BOT_TOKEN`,
  `AWS_ACCESS_KEY_ID`, …) each tagged with `{present: true|false, service: ...}`.
- `detected_credential_env_vars` — any other env var whose name matches a
  credential-shaped pattern (`*_API_KEY`, `*_TOKEN`, `*_SECRET`, `*_KEY`,
  `*_PASSWORD`, `*_CREDENTIALS`). Use this to find keys the curated list
  didn't already know about.

Call it before authoring a new skill to check what the LLM can actually
reach — no point writing a Slack poster if `SLACK_BOT_TOKEN` isn't set.

```
skill_search("selfmcp_env_keys")      # find the id
skill_execute(skill_id=<id>)          # no params needed
```

Inside a skill, values are still read the normal way via `os.environ` (the
subprocess inherits the server's full environment), so this skill is for
*discovery*, not for fetching secrets.

## Using API keys in skills

Skills run as Python subprocesses that **inherit the server's full environment**.
This means any environment variable (API key, secret, config) set on the server
is automatically available inside skill code via `os.environ` — no need to pass
keys as parameters.

### Reading a key inside a skill

```python
import json, os

params = json.loads(os.environ.get("SELFMCP_PARAMS", "{}"))
api_key = os.environ.get("ANTHROPIC_API_KEY")  # set once on the server, used everywhere
```

### Setting keys on the server

How you expose env vars to the server depends on your deployment:

| Deployment | How to set env vars |
|------------|---------------------|
| **Replit** | Tools → Secrets → add key/value pairs, then restart |
| **Local**  | `export ANTHROPIC_API_KEY=sk-ant-...` before running `server.py` |
| **Docker** | Pass `-e ANTHROPIC_API_KEY=sk-ant-...` to `docker run` |
| **Claude Desktop (stdio)** | Add to the `env` block in `claude_desktop_config.json` |

### Declaring `auth_config` (strongly recommended)

When creating a skill that calls an external API, always declare `auth_config`.
This lets `skill_execute` detect a missing key *before* running the skill and
return an actionable error instead of a cryptic 401:

```json
{
  "type": "api_key",
  "env_var": "ANTHROPIC_API_KEY",
  "instructions": "Get a key at https://console.anthropic.com/"
}
```

A complete `skill_create` call that follows this pattern:

```python
skill_create(
    name="write_haiku",
    description="Write a haiku on any topic using Claude.",
    body="""
import json, os
import anthropic

params = json.loads(os.environ.get("SELFMCP_PARAMS", "{}"))
topic = params.get("topic", "nature")

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env automatically
msg = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=64,
    messages=[{"role": "user", "content": f"Write a haiku about {topic}."}],
)
print(msg.content[0].text)
""",
    dependencies=["anthropic"],
    auth_config={
        "type": "api_key",
        "env_var": "ANTHROPIC_API_KEY",
        "instructions": "Get a key at https://console.anthropic.com/",
    },
)
```

When the key is missing, `skill_execute` returns:

```json
{
  "error": "missing_credentials",
  "missing": ["ANTHROPIC_API_KEY"],
  "hint": "Call skill_auth_url(skill_id=1) for setup instructions. On Replit, add the missing vars at: https://replit.com/@you/your-repl#secrets"
}
```

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
a pre-filled authorization URL, and on Replit a direct link to the Secrets
panel). The client renders that as a clickable link for the user.

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

5. **Add secrets** in **Tools → Secrets** (lock icon in the sidebar). Any key
   you add here becomes an environment variable available to the server _and_
   to every skill it executes:

   | Secret key          | Value                        | Purpose |
   |---------------------|------------------------------|---------|
   | `ANTHROPIC_API_KEY` | `sk-ant-...`                 | Required for skills that call Claude |
   | `OPENAI_API_KEY`    | `sk-...`                     | Enables real embeddings + OpenAI skills |
   | `SELFMCP_DB_PATH`   | `/home/user/selfmcp.db`      | Explicit persistent path (optional) |

   After adding a secret, **restart the Repl** so the server picks up the new
   value. Skills created after that point can use the key via `os.environ`.

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
      "env": {
        "SELFMCP_TRANSPORT": "stdio",
        "ANTHROPIC_API_KEY": "sk-ant-..."
      }
    }
  }
}
```

Or register as a remote MCP server in Claude.ai pointing at
`http://your-host:8000/mcp`.

## Troubleshooting

**Skill execution returns a 401 / auth error**

The skill is calling an external API but the key isn't set on the server.
Skills inherit the server's environment, so the fix is to add the key there:
- Replit: Tools → Secrets → add the key → restart the Repl
- Local: `export ANTHROPIC_API_KEY=sk-ant-...` then restart `server.py`

To make this detectable in the future, update the skill to declare
`auth_config` (see [Declaring auth_config](#declaring-auth_config-strongly-recommended)).

**`skill_create` returns a name-conflict error after deleting a skill**

`skill_delete` is a soft-delete — the row is kept for history. If you want to
reuse the same name, call `skill_create` again with the same name and it will
automatically reactivate the deleted entry rather than failing.

**Skills don't see environment variables I set**

Env vars are read when the server *starts*. If you add or change a secret
after the server is running, restart it. On Replit that means stopping and
re-running the Repl.

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
- **Soft-delete with name reuse.** `skill_delete` marks a row `is_active=0`
  and preserves history. `skill_create` with a previously-used name reactivates
  the old entry rather than failing — so a delete-then-recreate cycle works
  as expected.
- **Why SQLite + FTS5 rather than a vector DB?** Single file, no ops, deploys
  to Replit trivially. When you outgrow linear cosine sim, swap the
  `_embedding_upsert` / search implementation for a proper index — the
  rest of the code doesn't care.
- **Why SSE / streamable-HTTP rather than stdio by default?** The server
  persists state (the skill database). You want it as a long-lived process
  that multiple clients can share.
