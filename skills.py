"""Core skill registry logic.

Plain Python functions (no MCP decorators) so they're trivially testable
and reusable. ``server.py`` wraps these with ``@mcp.tool()``.

Everything that touches the database goes through ``get_conn`` and commits
inside the ``with`` block. The summary cache is regenerated lazily after
any write so that ``skill_list_summary`` is a cheap read.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from db import get_conn, pack_vector, unpack_vector
from embeddings import cosine_similarity, get_embedding
from executor import execute_skill as _execute_skill


# --------------------------------------------------------------------------- #
# Built-in self-documentation skill
# --------------------------------------------------------------------------- #

_ABOUT_SKILL_NAME = "selfmcp_about"

_ABOUT_SKILL_DESCRIPTION = (
    "Self-documentation for the selfMCP server. Returns structured facts about "
    "the source code location, interface, versioning, skill execution model, auth, "
    "all 8 bootstrap tools, search, and Replit setup. "
    "Pass {\"section\": \"<key>\"} to focus on one area, or omit params for the full doc. "
    "Available sections: source_code, interface, transport, persistence, versioning, "
    "execution, auth, bootstrap_tools, search, replit_setup."
)

_ABOUT_SKILL_BODY = '''import json, os

params = json.loads(os.environ.get("SELFMCP_PARAMS", "{}"))
section = params.get("section")

INFO = {
    "source_code": "https://github.com/yoheinakajima/selfMCP",
    "interface": (
        "MCP-only — there is no web UI. Connect via Claude.ai, Claude Desktop, "
        "Cursor, or any MCP-compatible client."
    ),
    "transport": (
        "streamable-http by default (mounts at /mcp), SSE, or stdio. "
        "Set SELFMCP_TRANSPORT env var to switch. "
        "Default port 8000; Replit injects PORT automatically."
    ),
    "persistence": (
        "Single SQLite file (selfmcp.db in the working directory). "
        "Survives restarts on Replit private Repls. "
        "Override location with SELFMCP_DB_PATH."
    ),
    "versioning": {
        "how_it_works": (
            "Every skill_update and skill_delete appends the prior row to "
            "skill_versions (append-only audit log). No history is ever deleted."
        ),
        "soft_delete": (
            "skill_delete sets is_active=0 on the row — it does NOT remove it. "
            "The name stays in the table but the skill is hidden from all normal queries."
        ),
        "name_reuse": (
            "skill_create with a previously-deleted name reactivates the old "
            "entry (archives the deleted state, bumps version, flips is_active=1). "
            "It does not error with a name-conflict."
        ),
        "rollback": (
            "Full history lives in skill_versions. Inspect it with skill_get_detail "
            "(shows current version number), then use skill_update to restore a prior body."
        ),
    },
    "execution": {
        "how": (
            "skill_execute writes the skill body to a temp directory and runs it "
            "as a Python subprocess. The temp dir is cleaned up after each run."
        ),
        "env_inheritance": (
            "The subprocess inherits the server\'s FULL environment. Any API key "
            "set on the server (e.g. ANTHROPIC_API_KEY added to Replit Secrets) "
            "is available inside skill code via os.environ — no need to pass keys "
            "as params."
        ),
        "params": (
            "Call skill_execute with a params dict. Inside the skill, read it with: "
            "params = json.loads(os.environ.get(\'SELFMCP_PARAMS\', \'{}\'))"
        ),
        "security": (
            "Not a hardened sandbox — skills run with server privileges in a "
            "fresh CWD. Wrap in a container or nsjail for untrusted code."
        ),
    },
    "auth": {
        "declare": (
            "Add auth_config to skill_create for any skill that reads an env-var key. "
            "skill_execute checks for the key before running; if missing it returns "
            "a missing_credentials error instead of a cryptic 401."
        ),
        "api_key_example": (
            \'{"type": "api_key", "env_var": "ANTHROPIC_API_KEY", \' +
            \'"instructions": "Get a key at https://console.anthropic.com/"}\'
        ),
        "replit_hint": (
            "When running on Replit, the missing_credentials error and skill_auth_url "
            "both include a direct link to the Secrets panel so the user can add the "
            "key without leaving Claude."
        ),
        "oauth2": (
            "Also supports type=oauth2 with auth_url, token_url, scopes, "
            "client_id_env, client_secret_env fields."
        ),
    },
    "bootstrap_tools": [
        "skill_create       — add a new skill; reactivates soft-deleted names automatically",
        "skill_update       — patch a skill by id and bump its version",
        "skill_delete       — soft-delete (row kept, history in skill_versions)",
        "skill_execute      — run a skill in a subprocess (inherits full server env)",
        "skill_list_summary — compact TOC [{id,name,short_description}] (cheap, safe to inject)",
        "skill_get_detail   — full record: body, deps, auth_config, version, timestamps",
        "skill_search       — hybrid FTS5+vector search; modes: keyword/vector/hybrid",
        "skill_auth_url     — auth instructions / prefilled OAuth2 URL for a skill",
    ],
    "search": {
        "modes": "keyword (FTS5 BM25), vector (cosine sim on stored embeddings), hybrid (default, 50/50 merge)",
        "embeddings": (
            "LiteLLM text-embedding-3-small by default (needs OPENAI_API_KEY). "
            "Falls back to a deterministic hash-based 256-dim embedding when offline."
        ),
        "workflow": "skill_list_summary → skill_search → skill_get_detail → skill_execute",
    },
    "replit_setup": [
        "1. Import the repo from GitHub (yoheinakajima/selfMCP) into Replit.",
        "2. Press Run — .replit sets transport and port automatically.",
        "3. Add secrets in Tools → Secrets: ANTHROPIC_API_KEY, OPENAI_API_KEY.",
        "4. Restart the Repl so the new secrets are picked up by the server.",
        "5. Connect Claude.ai to https://<repl-name>.<username>.repl.co/mcp",
    ],
}

if section:
    val = INFO.get(section)
    if val is None:
        print(json.dumps({"error": f"Unknown section \'{section}\'", "available": list(INFO.keys())}))
    else:
        print(json.dumps({section: val}, indent=2))
else:
    print(json.dumps(INFO, indent=2))
'''


def seed_about_skill() -> None:
    """Ensure the built-in selfmcp_about skill exists in the registry.

    Called once at server startup. Creates the skill only if no row
    (active or deleted) with the name already exists, so a user who
    intentionally deletes it won't have it resurrected on every restart.
    """
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id FROM skills WHERE name = ?", (_ABOUT_SKILL_NAME,)
        ).fetchone()
    if row is not None:
        return  # Already exists (active or deleted) — leave it alone
    skill_create(_ABOUT_SKILL_NAME, _ABOUT_SKILL_DESCRIPTION, _ABOUT_SKILL_BODY)


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _fts_upsert(conn, skill_id: int, name: str, description: str) -> None:
    conn.execute("DELETE FROM skills_fts WHERE rowid = ?", (skill_id,))
    conn.execute(
        "INSERT INTO skills_fts (rowid, name, description) VALUES (?, ?, ?)",
        (skill_id, name, description),
    )


def _embedding_upsert(conn, skill_id: int, text: str) -> None:
    vec, model = get_embedding(text)
    blob = pack_vector(vec)
    conn.execute(
        "INSERT INTO skill_embeddings (skill_id, vector, dim, model) "
        "VALUES (?, ?, ?, ?) "
        "ON CONFLICT(skill_id) DO UPDATE SET "
        "    vector = excluded.vector, dim = excluded.dim, model = excluded.model",
        (skill_id, blob, len(vec), model),
    )


def _archive_version(conn, row) -> None:
    conn.execute(
        "INSERT INTO skill_versions "
        "(skill_id, version, name, description, body, dependencies_json, auth_config_json, changed_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            row["id"],
            row["version"],
            row["name"],
            row["description"],
            row["body"],
            row["dependencies_json"],
            row["auth_config_json"],
            time.time(),
        ),
    )


def _regen_summary_cache() -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, name, description FROM skills "
            "WHERE is_active = 1 ORDER BY id"
        ).fetchall()
        summary = [
            {
                "id": r["id"],
                "name": r["name"],
                "short_description": _shorten(r["description"]),
            }
            for r in rows
        ]
        conn.execute(
            "INSERT INTO skill_summary_cache (id, summary_json, updated_at) "
            "VALUES (1, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET "
            "    summary_json = excluded.summary_json, "
            "    updated_at = excluded.updated_at",
            (json.dumps(summary), time.time()),
        )
    return summary


def _shorten(text: str, limit: int = 160) -> str:
    text = (text or "").strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "\u2026"


def _sanitize_fts_query(query: str) -> str:
    """Turn a freeform query into a safe FTS5 MATCH expression.

    FTS5 has its own query syntax (``AND``, ``OR``, quotes, colons, etc.)
    and unescaped punctuation causes syntax errors. We keep alphanum +
    underscore + hyphen, quote each token, and OR them together.
    """
    tokens: list[str] = []
    for raw in query.split():
        clean = "".join(c for c in raw if c.isalnum() or c in "_-")
        if clean:
            tokens.append(f'"{clean}"')
    if not tokens:
        return '""'
    return " OR ".join(tokens)


def _check_auth_requirements(auth_config: dict[str, Any]) -> list[str]:
    """Return list of env var names required by ``auth_config`` but missing."""
    missing: list[str] = []
    t = auth_config.get("type")
    if t == "api_key":
        env = auth_config.get("env_var")
        if env and not os.environ.get(env):
            missing.append(env)
    elif t == "oauth2":
        for key in ("client_id_env", "client_secret_env"):
            env = auth_config.get(key)
            if env and not os.environ.get(env):
                missing.append(env)
    return missing


def _row_to_detail(row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "name": row["name"],
        "description": row["description"],
        "body": row["body"],
        "dependencies": json.loads(row["dependencies_json"] or "[]"),
        "auth_config": json.loads(row["auth_config_json"]) if row["auth_config_json"] else None,
        "version": row["version"],
        "is_active": bool(row["is_active"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


# --------------------------------------------------------------------------- #
# CRUD
# --------------------------------------------------------------------------- #

def skill_create(
    name: str,
    description: str,
    body: str,
    dependencies: list[str] | None = None,
    auth_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a new skill. Writes to skills, seeds skill_versions v1,
    upserts FTS + embeddings, and regenerates the summary cache.

    If a soft-deleted skill with the same name already exists, it is
    reactivated with the new content instead of failing with a UNIQUE
    constraint error.
    """
    if not name or not description or not body:
        return {"error": "name, description, and body are required"}

    now = time.time()
    deps_json = json.dumps(dependencies or [])
    auth_json = json.dumps(auth_config) if auth_config else None

    skill_id: int
    skill_version: int

    with get_conn() as conn:
        # Reactivate a soft-deleted entry rather than hitting the UNIQUE constraint.
        deleted = conn.execute(
            "SELECT * FROM skills WHERE name = ? AND is_active = 0", (name,)
        ).fetchone()

        if deleted:
            _archive_version(conn, deleted)
            skill_version = deleted["version"] + 1
            conn.execute(
                "UPDATE skills SET "
                "  description = ?, body = ?, dependencies_json = ?, "
                "  auth_config_json = ?, version = ?, is_active = 1, "
                "  created_at = ?, updated_at = ? "
                "WHERE id = ?",
                (description, body, deps_json, auth_json, skill_version, now, now, deleted["id"]),
            )
            skill_id = deleted["id"]
        else:
            try:
                cur = conn.execute(
                    "INSERT INTO skills "
                    "(name, description, body, dependencies_json, auth_config_json, "
                    " version, is_active, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, 1, 1, ?, ?)",
                    (name, description, body, deps_json, auth_json, now, now),
                )
            except Exception as e:
                return {"error": f"create failed: {e}"}
            skill_id = cur.lastrowid
            skill_version = 1
            conn.execute(
                "INSERT INTO skill_versions "
                "(skill_id, version, name, description, body, dependencies_json, "
                " auth_config_json, changed_at) "
                "VALUES (?, 1, ?, ?, ?, ?, ?, ?)",
                (skill_id, name, description, body, deps_json, auth_json, now),
            )

        _fts_upsert(conn, skill_id, name, description)
        _embedding_upsert(conn, skill_id, f"{name}\n{description}")

    _regen_summary_cache()
    return {"id": skill_id, "name": name, "version": skill_version, "status": "created"}


def skill_update(
    skill_id: int,
    name: str | None = None,
    description: str | None = None,
    body: str | None = None,
    dependencies: list[str] | None = None,
    auth_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Patch an existing skill. Only provided fields are overwritten.
    Archives the prior row to skill_versions and bumps ``version``.
    """
    now = time.time()
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM skills WHERE id = ? AND is_active = 1", (skill_id,)
        ).fetchone()
        if not row:
            return {"error": f"skill {skill_id} not found"}

        _archive_version(conn, row)

        new_name = name if name is not None else row["name"]
        new_desc = description if description is not None else row["description"]
        new_body = body if body is not None else row["body"]
        new_deps = (
            json.dumps(dependencies) if dependencies is not None else row["dependencies_json"]
        )
        if auth_config is not None:
            new_auth = json.dumps(auth_config)
        else:
            new_auth = row["auth_config_json"]
        new_version = row["version"] + 1

        conn.execute(
            "UPDATE skills SET "
            "  name = ?, description = ?, body = ?, "
            "  dependencies_json = ?, auth_config_json = ?, "
            "  version = ?, updated_at = ? "
            "WHERE id = ?",
            (new_name, new_desc, new_body, new_deps, new_auth, new_version, now, skill_id),
        )
        _fts_upsert(conn, skill_id, new_name, new_desc)
        _embedding_upsert(conn, skill_id, f"{new_name}\n{new_desc}")

    _regen_summary_cache()
    return {
        "id": skill_id,
        "name": new_name,
        "version": new_version,
        "status": "updated",
    }


def skill_delete(skill_id: int) -> dict[str, Any]:
    """Soft-delete a skill. The row stays in ``skills`` with ``is_active=0``
    and the last active state is appended to ``skill_versions`` so you
    can still diff / restore.
    """
    now = time.time()
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM skills WHERE id = ? AND is_active = 1", (skill_id,)
        ).fetchone()
        if not row:
            return {"error": f"skill {skill_id} not found"}

        _archive_version(conn, row)
        conn.execute(
            "UPDATE skills SET is_active = 0, updated_at = ? WHERE id = ?",
            (now, skill_id),
        )
        conn.execute("DELETE FROM skills_fts WHERE rowid = ?", (skill_id,))
        conn.execute("DELETE FROM skill_embeddings WHERE skill_id = ?", (skill_id,))

    _regen_summary_cache()
    return {"id": skill_id, "status": "deleted"}


def skill_execute(
    skill_id: int,
    params: dict[str, Any] | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    """Run a skill's body in a sandboxed subprocess.

    Checks ``auth_config`` first — if required env vars aren't set, returns
    a ``missing_credentials`` error pointing at ``skill_auth_url`` instead
    of executing.
    """
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM skills WHERE id = ? AND is_active = 1", (skill_id,)
        ).fetchone()
        if not row:
            return {"error": f"skill {skill_id} not found"}
        body = row["body"]
        auth_config = json.loads(row["auth_config_json"]) if row["auth_config_json"] else None

    if auth_config:
        missing = _check_auth_requirements(auth_config)
        if missing:
            hint = f"Call skill_auth_url(skill_id={skill_id}) for setup instructions."
            repl_owner = os.environ.get("REPL_OWNER")
            repl_slug = os.environ.get("REPL_SLUG")
            if repl_owner and repl_slug:
                secrets_url = f"https://replit.com/@{repl_owner}/{repl_slug}#secrets"
                hint += f" On Replit, add the missing vars at: {secrets_url}"
            return {
                "error": "missing_credentials",
                "missing": missing,
                "hint": hint,
            }

    return _execute_skill(body, params or {}, timeout=timeout)


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #

def skill_list_summary() -> list[dict[str, Any]]:
    """Compact table of contents: ``[{id, name, short_description}, ...]``.

    Reads the materialized cache and regenerates it on miss. Cheap to
    inject into model context as a discovery index.
    """
    with get_conn() as conn:
        row = conn.execute(
            "SELECT summary_json FROM skill_summary_cache WHERE id = 1"
        ).fetchone()
        if row:
            return json.loads(row["summary_json"])
    return _regen_summary_cache()


def skill_get_detail(skill_id: int) -> dict[str, Any]:
    """Return the full skill record for ``skill_id``."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM skills WHERE id = ? AND is_active = 1", (skill_id,)
        ).fetchone()
        if not row:
            return {"error": f"skill {skill_id} not found"}
        return _row_to_detail(row)


def skill_search(
    query: str,
    mode: str = "hybrid",
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Find skills matching ``query``.

    ``mode``:
      - ``"keyword"``  FTS5 bm25 only
      - ``"vector"``   cosine similarity on stored embeddings only
      - ``"hybrid"``   both, merged with a 0.5/0.5 weighted sum (default)
    """
    if top_k < 1:
        top_k = 5
    if mode not in ("keyword", "vector", "hybrid"):
        mode = "hybrid"

    with get_conn() as conn:
        active_rows = conn.execute(
            "SELECT id, name, description FROM skills WHERE is_active = 1"
        ).fetchall()
        active: dict[int, dict[str, Any]] = {
            r["id"]: {
                "id": r["id"],
                "name": r["name"],
                "description": r["description"],
            }
            for r in active_rows
        }
        if not active:
            return []

        fts_scores: dict[int, float] = {}
        vec_scores: dict[int, float] = {}

        if mode in ("keyword", "hybrid") and query.strip():
            try:
                safe = _sanitize_fts_query(query)
                rows = conn.execute(
                    "SELECT rowid, bm25(skills_fts) AS score "
                    "FROM skills_fts WHERE skills_fts MATCH ? "
                    "ORDER BY score LIMIT 50",
                    (safe,),
                ).fetchall()
                if rows:
                    scores = [r["score"] for r in rows]
                    lo, hi = min(scores), max(scores)
                    span = (hi - lo) or 1.0
                    for r in rows:
                        sid = r["rowid"]
                        if sid in active:
                            # bm25: lower is better; invert to 0..1 where higher = better
                            fts_scores[sid] = 1.0 - ((r["score"] - lo) / span)
            except Exception:
                # Malformed query tokens shouldn't break vector search
                pass

        if mode in ("vector", "hybrid"):
            qvec, _ = get_embedding(query)
            emb_rows = conn.execute(
                "SELECT skill_id, vector, dim FROM skill_embeddings"
            ).fetchall()
            for r in emb_rows:
                sid = r["skill_id"]
                if sid not in active:
                    continue
                svec = unpack_vector(r["vector"], r["dim"])
                if len(svec) != len(qvec):
                    continue
                vec_scores[sid] = cosine_similarity(qvec, svec)

    if mode == "keyword":
        merged = fts_scores
    elif mode == "vector":
        merged = vec_scores
    else:
        all_ids = set(fts_scores) | set(vec_scores)
        merged = {
            sid: 0.5 * fts_scores.get(sid, 0.0) + 0.5 * vec_scores.get(sid, 0.0)
            for sid in all_ids
        }

    ranked = sorted(merged.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return [
        {
            "id": active[sid]["id"],
            "name": active[sid]["name"],
            "description": active[sid]["description"],
            "score": round(score, 4),
        }
        for sid, score in ranked
    ]


def skill_auth_url(skill_id: int) -> dict[str, Any]:
    """Return auth instructions / URL for a skill that declares ``auth_config``.

    The MCP protocol doesn't have a native "open this URL in the user's
    browser" primitive, so the client (Claude.ai, Cursor, …) is expected
    to render the returned ``auth_url`` or ``instructions`` as a clickable
    link / readable message.
    """
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM skills WHERE id = ? AND is_active = 1", (skill_id,)
        ).fetchone()
        if not row:
            return {"error": f"skill {skill_id} not found"}
        if not row["auth_config_json"]:
            return {
                "id": skill_id,
                "auth_required": False,
                "message": "This skill requires no authentication.",
            }
        auth_config = json.loads(row["auth_config_json"])

    missing = _check_auth_requirements(auth_config)
    t = auth_config.get("type")

    if t == "api_key":
        env_var = auth_config.get("env_var")
        instructions = auth_config.get(
            "instructions",
            f"Set the {env_var} environment variable and restart the server.",
        )
        result: dict[str, Any] = {
            "id": skill_id,
            "auth_type": "api_key",
            "env_var": env_var,
            "is_configured": not missing,
            "missing": missing,
            "instructions": instructions,
        }
        # On Replit, surface a direct link to the Secrets panel.
        repl_owner = os.environ.get("REPL_OWNER")
        repl_slug = os.environ.get("REPL_SLUG")
        if repl_owner and repl_slug and missing:
            setup_url = f"https://replit.com/@{repl_owner}/{repl_slug}#secrets"
            result["setup_url"] = setup_url
            result["instructions"] = (
                f"{instructions} "
                f"On Replit, open Secrets and add {env_var}: {setup_url}"
            )
        return result

    if t == "oauth2":
        auth_url = auth_config.get("auth_url", "")
        client_id = os.environ.get(
            auth_config.get("client_id_env", ""), "<CLIENT_ID>"
        )
        scopes = " ".join(auth_config.get("scopes", []))
        redirect_uri = auth_config.get("redirect_uri", "http://localhost:8000/oauth/callback")
        full_url = (
            f"{auth_url}?client_id={client_id}"
            f"&redirect_uri={redirect_uri}"
            f"&response_type=code"
            f"&scope={scopes}"
        )
        return {
            "id": skill_id,
            "auth_type": "oauth2",
            "is_configured": not missing,
            "missing": missing,
            "auth_url": full_url,
            "instructions": (
                "Visit the auth URL to authorize, then set the required env vars: "
                f"{missing or 'already configured'}."
            ),
        }

    # Unknown/custom auth type: return config so the client can decide.
    return {
        "id": skill_id,
        "auth_type": t,
        "is_configured": not missing,
        "missing": missing,
        "auth_config": auth_config,
    }
