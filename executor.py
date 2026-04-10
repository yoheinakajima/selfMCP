"""Sandboxed skill execution.

A skill's ``body`` is treated as a Python script — either raw Python or a
SKILL.md document containing a fenced ``python`` code block. The executor
writes it to a temp directory, runs it as a subprocess, and returns
``{stdout, stderr, exit_code}``.

Parameters are passed to the skill via the ``SELFMCP_PARAMS`` environment
variable as a JSON string. Skills can read it with::

    import json, os
    params = json.loads(os.environ.get("SELFMCP_PARAMS", "{}"))

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
