"""Tests for selfMCP core logic.

Runs against a temporary SQLite file per test — no MCP transport involved.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

# Redirect the DB path before importing modules that read it.
_TEST_DB = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_TEST_DB.close()
os.environ["SELFMCP_DB_PATH"] = _TEST_DB.name

import db  # noqa: E402
import skills  # noqa: E402
from embeddings import _local_embedding, cosine_similarity, get_embedding  # noqa: E402
from executor import execute_skill, extract_code  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_db(tmp_path, monkeypatch):
    """Point every module at an isolated per-test DB."""
    path = tmp_path / "test.db"
    monkeypatch.setattr(db, "DB_PATH", str(path))
    db.init_db(str(path))
    yield


# --------------------------------------------------------------------------- #
# CRUD
# --------------------------------------------------------------------------- #

def test_create_and_get_detail():
    res = skills.skill_create(
        name="hello",
        description="Prints a greeting to stdout",
        body="print('hi from skill')",
        dependencies=["requests"],
    )
    assert res["status"] == "created"
    assert res["version"] == 1
    sid = res["id"]

    detail = skills.skill_get_detail(sid)
    assert detail["name"] == "hello"
    assert detail["version"] == 1
    assert detail["dependencies"] == ["requests"]
    assert detail["auth_config"] is None
    assert "hi from skill" in detail["body"]


def test_create_validates_required_fields():
    res = skills.skill_create(name="", description="x", body="y")
    assert "error" in res


def test_update_bumps_version_and_archives_prior():
    sid = skills.skill_create("demo", "desc v1", "print(1)")["id"]
    res = skills.skill_update(sid, description="desc v2", body="print(2)")
    assert res["version"] == 2

    detail = skills.skill_get_detail(sid)
    assert detail["description"] == "desc v2"
    assert "print(2)" in detail["body"]

    # Prior version should be in skill_versions.
    with db.get_conn() as conn:
        rows = conn.execute(
            "SELECT version, description, body FROM skill_versions WHERE skill_id = ? ORDER BY version",
            (sid,),
        ).fetchall()
    versions = [(r["version"], r["description"], r["body"]) for r in rows]
    # We expect at least v1 archived; v1 seeded on create and v1 re-archived on update.
    assert any(v == 1 and "desc v1" in d and "print(1)" in b for (v, d, b) in versions)


def test_update_unknown_id_returns_error():
    res = skills.skill_update(9999, name="x")
    assert "error" in res


def test_delete_soft_deletes_and_regens_summary():
    sid = skills.skill_create("temp", "temporary skill", "print(3)")["id"]
    assert any(s["id"] == sid for s in skills.skill_list_summary())

    res = skills.skill_delete(sid)
    assert res["status"] == "deleted"
    assert not any(s["id"] == sid for s in skills.skill_list_summary())
    assert "error" in skills.skill_get_detail(sid)


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #

def test_list_summary_reflects_active_skills():
    a = skills.skill_create("alpha", "alpha description", "print('a')")["id"]
    b = skills.skill_create("beta", "beta description", "print('b')")["id"]
    summary = skills.skill_list_summary()
    ids = {s["id"] for s in summary}
    assert a in ids and b in ids
    for s in summary:
        assert "short_description" in s
        assert "name" in s


def test_search_keyword_mode_finds_fts_matches():
    skills.skill_create("slack_poster", "Post a message to a Slack channel", "print('slack')")
    skills.skill_create("gcal_reader", "Read events from Google Calendar", "print('cal')")
    results = skills.skill_search("slack", mode="keyword", top_k=5)
    assert results, "expected at least one keyword match"
    assert results[0]["name"] == "slack_poster"


def test_search_vector_mode_returns_ranked_results():
    skills.skill_create("pdf_extractor", "Extract text from a PDF file", "print('pdf')")
    skills.skill_create("csv_parser", "Parse a CSV into rows", "print('csv')")
    results = skills.skill_search("extract text from a document", mode="vector", top_k=2)
    assert len(results) >= 1
    for r in results:
        assert "score" in r


def test_search_hybrid_merges_results():
    skills.skill_create("image_resize", "Resize images on disk", "print('img')")
    skills.skill_create("thumbnail_maker", "Generate thumbnails from images", "print('thumb')")
    results = skills.skill_search("resize images", mode="hybrid", top_k=5)
    names = [r["name"] for r in results]
    assert "image_resize" in names


def test_search_handles_empty_query_gracefully():
    skills.skill_create("one", "foo", "print(1)")
    # empty query: vector still runs (hash on empty → zero vector OK), keyword skipped
    results = skills.skill_search("", mode="hybrid", top_k=3)
    assert isinstance(results, list)


def test_search_sanitizes_punctuation_in_query():
    skills.skill_create("xml_cleaner", "Clean up XML files", "print('x')")
    # Punctuation that would break raw FTS5 MATCH.
    results = skills.skill_search("xml: cleaner!?", mode="keyword", top_k=5)
    assert any(r["name"] == "xml_cleaner" for r in results)


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #

def test_extract_code_pulls_from_markdown():
    body = """# my skill
some prose
```python
print("hello")
```
more prose
"""
    assert extract_code(body).strip() == 'print("hello")'


def test_extract_code_passthrough_for_plain_python():
    body = "import sys\nprint(sys.version)"
    assert extract_code(body) == body


def test_execute_skill_returns_stdout():
    result = execute_skill("print('hello world')", {}, timeout=10)
    assert result["exit_code"] == 0
    assert "hello world" in result["stdout"]
    assert result["timed_out"] is False


def test_execute_skill_respects_params_env():
    body = (
        "import os, json\n"
        "p = json.loads(os.environ['SELFMCP_PARAMS'])\n"
        "print(p['greeting'], p['who'])\n"
    )
    result = execute_skill(body, {"greeting": "hi", "who": "claude"}, timeout=10)
    assert result["exit_code"] == 0
    assert "hi claude" in result["stdout"]


def test_execute_skill_times_out():
    body = "import time; time.sleep(5)"
    result = execute_skill(body, {}, timeout=1)
    assert result["timed_out"] is True
    assert result["exit_code"] == -1


def test_skill_execute_via_registry():
    sid = skills.skill_create("echo", "echo a param", body=(
        "import os, json\n"
        "print(json.loads(os.environ['SELFMCP_PARAMS'])['msg'])\n"
    ))["id"]
    out = skills.skill_execute(sid, params={"msg": "roundtrip"}, timeout=10)
    assert out["exit_code"] == 0
    assert "roundtrip" in out["stdout"]


# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #

def test_auth_url_api_key_missing(monkeypatch):
    monkeypatch.delenv("DEMO_API_KEY", raising=False)
    sid = skills.skill_create(
        "auth_demo",
        "needs an api key",
        "print('x')",
        auth_config={
            "type": "api_key",
            "env_var": "DEMO_API_KEY",
            "instructions": "Get one at https://example.com",
        },
    )["id"]

    info = skills.skill_auth_url(sid)
    assert info["auth_type"] == "api_key"
    assert info["is_configured"] is False
    assert info["missing"] == ["DEMO_API_KEY"]
    assert "example.com" in info["instructions"]


def test_auth_url_api_key_configured(monkeypatch):
    monkeypatch.setenv("DEMO_API_KEY", "set")
    sid = skills.skill_create(
        "auth_demo2",
        "needs an api key",
        "print('x')",
        auth_config={"type": "api_key", "env_var": "DEMO_API_KEY"},
    )["id"]
    info = skills.skill_auth_url(sid)
    assert info["is_configured"] is True
    assert info["missing"] == []


def test_auth_url_oauth2_returns_prefilled_url(monkeypatch):
    monkeypatch.setenv("DEMO_CLIENT_ID", "abc123")
    monkeypatch.delenv("DEMO_CLIENT_SECRET", raising=False)
    sid = skills.skill_create(
        "oauth_demo",
        "oauth2 skill",
        "print('x')",
        auth_config={
            "type": "oauth2",
            "auth_url": "https://example.com/auth",
            "token_url": "https://example.com/token",
            "scopes": ["read", "write"],
            "client_id_env": "DEMO_CLIENT_ID",
            "client_secret_env": "DEMO_CLIENT_SECRET",
        },
    )["id"]
    info = skills.skill_auth_url(sid)
    assert info["auth_type"] == "oauth2"
    assert "abc123" in info["auth_url"]
    assert "DEMO_CLIENT_SECRET" in info["missing"]
    assert info["is_configured"] is False


def test_execute_blocked_when_auth_missing(monkeypatch):
    monkeypatch.delenv("MISSING_KEY", raising=False)
    sid = skills.skill_create(
        "gated",
        "gated skill",
        "print('should not run')",
        auth_config={"type": "api_key", "env_var": "MISSING_KEY"},
    )["id"]
    res = skills.skill_execute(sid, {}, timeout=5)
    assert res.get("error") == "missing_credentials"
    assert "MISSING_KEY" in res["missing"]


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #

def test_local_embedding_is_deterministic_and_normalized():
    v1 = _local_embedding("hello world")
    v2 = _local_embedding("hello world")
    assert v1 == v2
    import math
    norm = math.sqrt(sum(x * x for x in v1))
    assert abs(norm - 1.0) < 1e-5


def test_cosine_similarity_bounds():
    v = _local_embedding("greetings, earthlings")
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-5
    assert cosine_similarity(v, [0.0] * len(v)) == 0.0


def test_get_embedding_returns_vector_and_model():
    vec, model = get_embedding("hello")
    assert isinstance(vec, list) and vec
    assert isinstance(model, str) and model
