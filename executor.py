"""Sandboxed skill execution.

A skill's ``body`` is treated as a Python script — either raw Python or a
SKILL.md document containing a fenced ``python`` code block. The executor
writes it to a temp directory, runs it as a subprocess, and returns
``{stdout, stderr, exit_code}``.

Parameters are passed to the skill via the ``SELFMCP_PARAMS`` environment
variable as a JSON string. Skills can read it with::

    import json, os
    params = json.loads(os.environ.get("SELFMCP_PARAMS", "{}"))

Skill composition
-----------------
The executor also exposes the in-skill SDK (:mod:`selfmcp_sdk`) to every
subprocess by:

  * prepending the selfMCP repo root to the child's ``PYTHONPATH`` so
    ``import selfmcp_sdk`` works from the temp cwd, and
  * resolving ``SELFMCP_DB_PATH`` to an absolute path so the SDK can open
    the same SQLite file the server is using.

That gives skill code first-class access to ``run_skill`` /
``search_skills`` / ``get_skill`` / ``list_skills`` for composing other
skills without going back through the MCP client.

This is not a hardened sandbox — skills run with the server's privileges
in a fresh CWD. Run untrusted skills inside a container or a nsjail-style
wrapper if you need stronger isolation.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any

_CODE_BLOCK_RE = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.DOTALL)

# Absolute path to the directory containing this file (the selfMCP repo
# root). Used to expose ``selfmcp_sdk`` and friends on the child's
# PYTHONPATH so skills can compose each other.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def extract_code(body: str) -> str:
    """Pull a Python code block out of a SKILL.md body if present.

    If the body contains one or more ```python fenced blocks, concatenate
    them and return that. Otherwise return the body verbatim (assumed to
    be a plain Python script).
    """
    blocks = _CODE_BLOCK_RE.findall(body)
    if blocks:
        return "\n\n".join(b.rstrip() for b in blocks)
    return body


def execute_skill(
    body: str,
    params: dict[str, Any] | None = None,
    timeout: int = 30,
    env_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run a skill body in a subprocess and return its output."""
    params = params or {}
    code = extract_code(body)

    workdir = tempfile.mkdtemp(prefix="selfmcp_skill_")
    try:
        script_path = os.path.join(workdir, "skill.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(code)

        env = os.environ.copy()
        env["SELFMCP_PARAMS"] = json.dumps(params)

        # Make the DB path absolute so the in-skill SDK can find the same
        # database the server is using even though the subprocess runs
        # from a temp cwd. Resolution happens against the *server's* cwd
        # (i.e. this process), which is the right anchor for relative
        # paths like the default "selfmcp.db".
        db_path = env.get("SELFMCP_DB_PATH", "selfmcp.db")
        if not os.path.isabs(db_path):
            db_path = os.path.abspath(db_path)
        env["SELFMCP_DB_PATH"] = db_path

        # Expose the in-skill SDK (selfmcp_sdk, executor, db, ...) on the
        # child's PYTHONPATH so a skill body can do `from selfmcp_sdk
        # import run_skill` and compose other skills.
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            _REPO_ROOT + os.pathsep + existing_pp if existing_pp else _REPO_ROOT
        )

        if env_overrides:
            env.update(env_overrides)

        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                cwd=workdir,
                env=env,
                timeout=timeout,
            )
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode,
                "timed_out": False,
            }
        except subprocess.TimeoutExpired as exc:
            return {
                "stdout": exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
                "stderr": (
                    (exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or ""))
                    + f"\n[selfmcp] execution timed out after {timeout}s"
                ),
                "exit_code": -1,
                "timed_out": True,
            }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
