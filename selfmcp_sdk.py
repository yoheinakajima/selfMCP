"""In-skill SDK for composing selfMCP skills.

This module is automatically importable from inside any skill executed via
``skill_execute``. It lets a skill discover, inspect, and invoke *other*
skills without going back through the MCP server / client round trip.

It works because :func:`executor.execute_skill` injects the selfMCP repo
root into the child subprocess's ``PYTHONPATH`` and forwards an absolute
``SELFMCP_DB_PATH`` so the SDK can open the same SQLite database the
server is using.

Typical usage inside a skill body::

    import json, os
    from selfmcp_sdk import run_skill, search_skills

    # Find a candidate sub-skill by keyword.
    hits = search_skills("slack post message")
    if not hits:
        raise SystemExit("no slack-poster skill found")

    # Invoke it. Sub-skills receive their own SELFMCP_PARAMS just like a
    # top-level skill_execute call.
    result = run_skill(
        hits[0]["id"],
        params={"channel": "#general", "text": "hello from a parent skill"},
    )
    if result["exit_code"] != 0:
        raise SystemExit(result["stderr"])
    print(result["stdout"])

The SDK is intentionally tiny and dependency-free (only ``sqlite3`` and
``json`` from the stdlib, plus a lazy import of :mod:`executor`) so a
skill subprocess pays no extra import cost unless it actually composes.

Sub-skill execution is recursive: a skill called via :func:`run_skill`
can itself call :func:`run_skill`, since each invocation spawns a fresh
subprocess with the same PYTHONPATH/DB env wired up by the executor.
There is no built-in recursion limit; if you build a chain that loops,
the OS process tree (and your timeout) is what stops it.
"""

from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Union

# Resolved lazily so importing this module never touches the filesystem.
_DB_PATH_ENV = "SELFMCP_DB_PATH"


SkillRef = Union[int, str]


def _db_path() -> str:
    return os.environ.get(_DB_PATH_ENV, "selfmcp.db")


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    return conn


def _resolve(name_or_id: SkillRef) -> sqlite3.Row | None:
    """Look up an active skill row by id or name."""
    with _conn() as conn:
        if isinstance(name_or_id, int) or (
            isinstance(name_or_id, str) and name_or_id.isdigit()
        ):
            row = conn.execute(
                "SELECT * FROM skills WHERE id = ? AND is_active = 1",
                (int(name_or_id),),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM skills WHERE name = ? AND is_active = 1",
                (str(name_or_id),),
            ).fetchone()
    return row


def _missing_credentials(auth_config: dict[str, Any]) -> list[str]:
    """Return env-var names declared by ``auth_config`` that are unset."""
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


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #

def list_skills() -> list[dict[str, Any]]:
    """Return ``[{id, name, description}]`` for every active skill.

    Mirrors the server-side ``skill_list_summary`` tool but reads the DB
    directly so it works from inside a skill subprocess.
    """
    with _conn() as conn:
        rows = conn.execute(
            "SELECT id, name, description FROM skills "
            "WHERE is_active = 1 ORDER BY id"
        ).fetchall()
    return [
        {"id": r["id"], "name": r["name"], "description": r["description"]}
        for r in rows
    ]


def search_skills(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Keyword (FTS5) search over the skill registry.

    Returns ``[{id, name, description}]`` ranked by BM25, best first.
    Punctuation is stripped from ``query`` to keep FTS5's MATCH grammar
    happy. This is intentionally keyword-only — vector search would pull
    in the embeddings stack and force every sub-skill subprocess to
    initialize LiteLLM, which is overkill for in-skill discovery.
    """
    if top_k < 1:
        top_k = 5

    tokens: list[str] = []
    for raw in (query or "").split():
        clean = "".join(ch for ch in raw if ch.isalnum() or ch in "_-")
        if clean:
            tokens.append(f'"{clean}"')
    if not tokens:
        return []
    match_expr = " OR ".join(tokens)

    with _conn() as conn:
        try:
            rows = conn.execute(
                "SELECT s.id, s.name, s.description "
                "FROM skills_fts f "
                "JOIN skills s ON s.id = f.rowid "
                "WHERE skills_fts MATCH ? AND s.is_active = 1 "
                "ORDER BY bm25(skills_fts) LIMIT ?",
                (match_expr, top_k),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
    return [
        {"id": r["id"], "name": r["name"], "description": r["description"]}
        for r in rows
    ]


def get_skill(name_or_id: SkillRef) -> dict[str, Any] | None:
    """Return the full record for one skill, or ``None`` if not found.

    Use this to inspect another skill's body before deciding whether to
    invoke it — handy for debugging composition or for skills that want
    to wrap / template another skill's output.
    """
    row = _resolve(name_or_id)
    if not row:
        return None
    return {
        "id": row["id"],
        "name": row["name"],
        "description": row["description"],
        "body": row["body"],
        "dependencies": json.loads(row["dependencies_json"] or "[]"),
        "auth_config": (
            json.loads(row["auth_config_json"]) if row["auth_config_json"] else None
        ),
        "version": row["version"],
    }


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #

def run_skill(
    name_or_id: SkillRef,
    params: dict[str, Any] | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    """Execute another skill in a fresh subprocess and return its output.

    Returns the same shape as the server-side ``skill_execute`` tool::

        {"stdout": str, "stderr": str, "exit_code": int, "timed_out": bool}

    or, on lookup / auth failure::

        {"error": "skill_not_found", "ref": <name_or_id>}
        {"error": "missing_credentials", "missing": [<env_var>, ...]}

    Auth handling mirrors the server: if the target skill declares an
    ``auth_config`` whose env vars are unset on the current process, the
    sub-call short-circuits with ``missing_credentials`` instead of
    running the body and producing a cryptic 401.

    The sub-skill receives ``params`` via its own ``SELFMCP_PARAMS`` env
    var (the parent's ``SELFMCP_PARAMS`` is overwritten for the child),
    and inherits the rest of the parent's environment — including any
    API keys the server set.
    """
    row = _resolve(name_or_id)
    if not row:
        return {"error": "skill_not_found", "ref": name_or_id}

    if row["auth_config_json"]:
        auth_config = json.loads(row["auth_config_json"])
        missing = _missing_credentials(auth_config)
        if missing:
            return {"error": "missing_credentials", "missing": missing}

    # Lazy import: keeps `import selfmcp_sdk` side-effect-free for skills
    # that only want discovery helpers and don't actually invoke anything.
    from executor import execute_skill as _execute_skill

    return _execute_skill(row["body"], params or {}, timeout=timeout)


__all__ = [
    "list_skills",
    "search_skills",
    "get_skill",
    "run_skill",
]
