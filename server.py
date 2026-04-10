"""FastMCP entry point for selfMCP.

Registers the eight bootstrap tools on a FastMCP server and runs it
over streamable HTTP (default), SSE, or stdio depending on the
``SELFMCP_TRANSPORT`` environment variable.

Environment variables
---------------------
SELFMCP_DB_PATH       Path to the SQLite file (default: ./selfmcp.db)
SELFMCP_TRANSPORT     stdio | sse | streamable-http  (default: streamable-http)
SELFMCP_HOST          Bind host when using HTTP/SSE  (default: 0.0.0.0)
SELFMCP_PORT          Bind port when using HTTP/SSE  (default: 8000)
SELFMCP_EMBED_MODEL   LiteLLM model for embeddings (default: text-embedding-3-small)
SELFMCP_USE_LITELLM   Force LiteLLM even without OPENAI_API_KEY (truthy value)
OPENAI_API_KEY        Enables LiteLLM remote embeddings
"""

from __future__ import annotations

import os
from typing import Any

from mcp.server.fastmcp import FastMCP

import skills as S
from db import init_db
from skills import seed_core_skills


def build_server() -> FastMCP:
    mcp = FastMCP(
        "selfmcp",
        instructions=(
            "selfMCP — a self-extending skill registry. "
            "Source: https://github.com/yoheinakajima/selfMCP. "
            "Interface: MCP-only, no web UI; connect via Claude.ai, Claude Desktop, Cursor, etc. "
            "Versioning: skill_delete is a soft-delete (history kept in skill_versions); "
            "skill_create reactivates a deleted name instead of erroring. "
            "API keys: skills inherit the server's full environment, so keys set on the server "
            "(e.g. ANTHROPIC_API_KEY in Replit Secrets) are available via os.environ inside "
            "skill code — never pass keys as params; declare auth_config so skill_execute "
            "can detect a missing key before running. "
            "Workflow: skill_list_summary → skill_search → skill_get_detail → skill_execute. "
            "Core skills (always present, undeletable): "
            "'selfmcp_about' — execute it for detailed self-documentation about "
            "versioning, execution, auth, all 8 tools, and setup. "
            "'selfmcp_env_keys' — execute it to discover which API keys / credential "
            "env vars are currently available on the server (names only, never values), "
            "so you know which external APIs you can call before authoring skills."
        ),
        host=os.environ.get("SELFMCP_HOST", "0.0.0.0"),
        port=int(os.environ.get("SELFMCP_PORT", os.environ.get("PORT", "8000"))),
    )

    # ------------------------------------------------------------------ CRUD
    @mcp.tool()
    def skill_create(
        name: str,
        description: str,
        body: str,
        dependencies: list[str] | None = None,
        auth_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a new skill.

        Args:
            name: Unique short identifier for the skill.
            description: One-paragraph description used for search + summary.
            body: The SKILL.md content or raw Python script to store.
                Skills run as subprocesses and inherit the server's environment,
                so any API key in the server env (e.g. ANTHROPIC_API_KEY) is
                available via os.environ — never pass keys as params.
            dependencies: Optional list of pip package specs the skill needs.
            auth_config: Declare this for any skill that reads an env-var API key.
                skill_execute will check for the key before running and surface
                setup instructions if it is missing. Supported types:
                {"type": "api_key", "env_var": "ANTHROPIC_API_KEY",
                 "instructions": "Get a key at https://console.anthropic.com/"}
                {"type": "oauth2", "auth_url": "...", "token_url": "...",
                 "scopes": [...], "client_id_env": "...", "client_secret_env": "..."}

        Returns:
            {"id", "name", "version", "status"} or {"error": "..."}.
            If a soft-deleted skill with the same name exists it is reactivated
            rather than returning a name-conflict error.
        """
        return S.skill_create(name, description, body, dependencies, auth_config)

    @mcp.tool()
    def skill_update(
        skill_id: int,
        name: str | None = None,
        description: str | None = None,
        body: str | None = None,
        dependencies: list[str] | None = None,
        auth_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Update an existing skill. Only provided fields change. Bumps version
        and archives the prior state to skill_versions for diff/rollback."""
        return S.skill_update(
            skill_id, name, description, body, dependencies, auth_config
        )

    @mcp.tool()
    def skill_delete(skill_id: int) -> dict[str, Any]:
        """Soft-delete a skill by id. History remains in skill_versions."""
        return S.skill_delete(skill_id)

    @mcp.tool()
    def skill_execute(
        skill_id: int,
        params: dict[str, Any] | None = None,
        timeout: int = 30,
    ) -> dict[str, Any]:
        """Execute a skill in a sandboxed subprocess.

        ``params`` is forwarded to the skill as JSON in the ``SELFMCP_PARAMS``
        env var. The subprocess also inherits the server's full environment, so
        API keys configured on the server (e.g. ANTHROPIC_API_KEY set in Replit
        Secrets) are available to skill code via os.environ without any extra
        plumbing.

        Returns ``{stdout, stderr, exit_code, timed_out}`` on success, or a
        ``missing_credentials`` error with setup instructions if the skill's
        auth_config declares an env var that isn't set on the server.
        """
        return S.skill_execute(skill_id, params, timeout)

    # ------------------------------------------------------------- Discovery
    @mcp.tool()
    def skill_list_summary() -> list[dict[str, Any]]:
        """Return a compact list of every active skill as
        ``[{id, name, short_description}]``. This is the skill registry's
        table of contents — cheap to inject into context as the first step
        of discovery.
        """
        return S.skill_list_summary()

    @mcp.tool()
    def skill_get_detail(skill_id: int) -> dict[str, Any]:
        """Return the full skill record: body, dependencies, auth_config,
        version, timestamps. Call this after skill_search identifies a
        candidate — keeps context lean by loading bodies only on demand."""
        return S.skill_get_detail(skill_id)

    @mcp.tool()
    def skill_search(
        query: str,
        mode: str = "hybrid",
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Search the skill registry.

        Args:
            query: Natural-language query.
            mode: "keyword" (FTS5 bm25), "vector" (cosine similarity on
                stored embeddings), or "hybrid" (both, merged). Default hybrid.
            top_k: Max results. Default 5.

        Returns:
            ``[{id, name, description, score}]`` sorted high→low.
        """
        return S.skill_search(query, mode, top_k)

    @mcp.tool()
    def skill_auth_url(skill_id: int) -> dict[str, Any]:
        """Return the auth URL / setup instructions for a skill with
        ``auth_config``. The calling client is expected to render the
        returned URL as a clickable link for the user."""
        return S.skill_auth_url(skill_id)

    return mcp


def main() -> None:
    init_db()
    seed_core_skills()
    transport = os.environ.get("SELFMCP_TRANSPORT", "streamable-http")
    if transport not in ("stdio", "sse", "streamable-http"):
        raise SystemExit(
            f"Invalid SELFMCP_TRANSPORT={transport!r}; "
            "must be one of stdio, sse, streamable-http"
        )
    mcp = build_server()
    mcp.run(transport=transport)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
