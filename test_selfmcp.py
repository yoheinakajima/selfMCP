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
    """Point every module at an isolated per-test DB.

    Also exports ``SELFMCP_DB_PATH`` so subprocesses spawned by
    :mod:`executor` (and the in-skill ``selfmcp_sdk``) read from the
    same per-test database instead of the package-global one set at
    import time.
    """
    path = tmp_path / "test.db"
    monkeypatch.setattr(db, "DB_PATH", str(path))
    monkeypatch.setenv("SELFMCP_DB_PATH", str(path))
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


# --------------------------------------------------------------------------- #
# Core skills (undeletable built-ins)
# --------------------------------------------------------------------------- #

def test_seed_core_skills_creates_both_builtins():
    skills.seed_core_skills()
    summary = skills.skill_list_summary()
    names = {s["name"] for s in summary}
    assert "selfmcp_about" in names
    assert "selfmcp_env_keys" in names


def test_seed_core_skills_is_idempotent():
    skills.seed_core_skills()
    first = skills.skill_list_summary()
    skills.seed_core_skills()
    second = skills.skill_list_summary()
    # Same ids, no duplicates created on second call.
    assert [s["id"] for s in first] == [s["id"] for s in second]


def test_core_skill_cannot_be_deleted():
    skills.seed_core_skills()
    about = next(
        s for s in skills.skill_list_summary() if s["name"] == "selfmcp_about"
    )
    res = skills.skill_delete(about["id"])
    assert res.get("error") == "cannot_delete_core_skill"
    # Still in the summary — not actually deleted.
    assert any(
        s["name"] == "selfmcp_about" for s in skills.skill_list_summary()
    )


def test_env_keys_skill_cannot_be_deleted():
    skills.seed_core_skills()
    env_skill = next(
        s for s in skills.skill_list_summary() if s["name"] == "selfmcp_env_keys"
    )
    res = skills.skill_delete(env_skill["id"])
    assert res.get("error") == "cannot_delete_core_skill"
    assert any(
        s["name"] == "selfmcp_env_keys" for s in skills.skill_list_summary()
    )


def test_core_skill_cannot_be_renamed_but_body_can_be_updated():
    skills.seed_core_skills()
    about = next(
        s for s in skills.skill_list_summary() if s["name"] == "selfmcp_about"
    )
    # Rename is rejected.
    rename_res = skills.skill_update(about["id"], name="custom_about")
    assert rename_res.get("error") == "cannot_rename_core_skill"
    # Name unchanged after the failed rename.
    detail = skills.skill_get_detail(about["id"])
    assert detail["name"] == "selfmcp_about"
    # Body edits ARE allowed.
    body_res = skills.skill_update(about["id"], body="print('custom body')")
    assert body_res.get("status") == "updated"
    assert "custom body" in skills.skill_get_detail(about["id"])["body"]


def test_seed_core_skills_reactivates_legacy_soft_deleted_row():
    # Simulate a legacy DB where the about skill was deleted before the
    # undeletable guard existed: insert a row directly with is_active=0.
    import time as _t
    with db.get_conn() as conn:
        conn.execute(
            "INSERT INTO skills (name, description, body, dependencies_json, "
            "auth_config_json, version, is_active, created_at, updated_at) "
            "VALUES (?, ?, ?, '[]', NULL, 1, 0, ?, ?)",
            ("selfmcp_about", "legacy deleted", "print('old')", _t.time(), _t.time()),
        )
    skills.seed_core_skills()
    names = {s["name"] for s in skills.skill_list_summary()}
    assert "selfmcp_about" in names
    assert "selfmcp_env_keys" in names


def test_env_keys_skill_runs_and_reports_known_services(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-xyz")
    monkeypatch.setenv("CUSTOM_SERVICE_API_KEY", "abc123")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    skills.seed_core_skills()
    env_skill = next(
        s for s in skills.skill_list_summary() if s["name"] == "selfmcp_env_keys"
    )
    result = skills.skill_execute(env_skill["id"], params={}, timeout=15)
    assert result["exit_code"] == 0, result.get("stderr")
    payload = json.loads(result["stdout"])

    # Curated list shows correct present/absent status.
    assert payload["known_services"]["ANTHROPIC_API_KEY"]["present"] is True
    assert payload["known_services"]["OPENAI_API_KEY"]["present"] is False

    # Detected list surfaces both well-known and unknown credential-shaped vars.
    assert "ANTHROPIC_API_KEY" in payload["detected_credential_env_vars"]
    assert "CUSTOM_SERVICE_API_KEY" in payload["detected_credential_env_vars"]
    assert "CUSTOM_SERVICE_API_KEY" in payload["additional_detected"]

    # The skill MUST never leak the actual secret value.
    assert "sk-ant-test-xyz" not in result["stdout"]
    assert "abc123" not in result["stdout"]

    # SELFMCP_* internals should not be flagged as credentials.
    assert not any(
        k.startswith("SELFMCP_") for k in payload["detected_credential_env_vars"]
    )


def test_is_core_skill_helper():
    assert skills.is_core_skill("selfmcp_about") is True
    assert skills.is_core_skill("selfmcp_env_keys") is True
    assert skills.is_core_skill("some_user_skill") is False


# --------------------------------------------------------------------------- #
# Skill composition (in-skill SDK)
# --------------------------------------------------------------------------- #

def test_sdk_list_skills_returns_active_rows():
    import selfmcp_sdk

    a = skills.skill_create("alpha_sdk", "alpha sdk skill", "print('a')")["id"]
    b = skills.skill_create("beta_sdk", "beta sdk skill", "print('b')")["id"]
    listed = selfmcp_sdk.list_skills()
    ids = {s["id"] for s in listed}
    assert a in ids and b in ids


def test_sdk_search_skills_keyword_match():
    import selfmcp_sdk

    skills.skill_create("notion_writer", "Write a page to Notion", "print('n')")
    skills.skill_create("calendar_reader", "Read events from Google Calendar", "print('c')")
    hits = selfmcp_sdk.search_skills("notion")
    assert hits and hits[0]["name"] == "notion_writer"


def test_sdk_search_skills_handles_punctuation_and_empty_query():
    import selfmcp_sdk

    skills.skill_create("xml_cleaner_sdk", "Clean up XML files", "print('x')")
    assert selfmcp_sdk.search_skills("") == []
    hits = selfmcp_sdk.search_skills("xml: cleaner!?")
    assert any(h["name"] == "xml_cleaner_sdk" for h in hits)


def test_sdk_get_skill_by_name_and_id():
    import selfmcp_sdk

    sid = skills.skill_create("inspect_me", "describe", "print('hi')")["id"]
    by_id = selfmcp_sdk.get_skill(sid)
    by_name = selfmcp_sdk.get_skill("inspect_me")
    assert by_id is not None and by_name is not None
    assert by_id["id"] == by_name["id"] == sid
    assert "print('hi')" in by_id["body"]
    assert selfmcp_sdk.get_skill("nope_not_here") is None


def test_sdk_run_skill_executes_target_in_subprocess():
    import selfmcp_sdk

    skills.skill_create(
        "inner_echo",
        "echo a SELFMCP_PARAMS message",
        "import os, json; print(json.loads(os.environ['SELFMCP_PARAMS'])['msg'])",
    )
    out = selfmcp_sdk.run_skill("inner_echo", params={"msg": "from-sdk"}, timeout=10)
    assert out["exit_code"] == 0
    assert "from-sdk" in out["stdout"]


def test_sdk_run_skill_returns_error_for_unknown_target():
    import selfmcp_sdk

    out = selfmcp_sdk.run_skill("does_not_exist", timeout=5)
    assert out.get("error") == "skill_not_found"


def test_sdk_run_skill_blocks_when_target_missing_credentials(monkeypatch):
    import selfmcp_sdk

    monkeypatch.delenv("SDK_DEMO_KEY", raising=False)
    skills.skill_create(
        "gated_inner",
        "needs SDK_DEMO_KEY",
        "print('should not run')",
        auth_config={"type": "api_key", "env_var": "SDK_DEMO_KEY"},
    )
    out = selfmcp_sdk.run_skill("gated_inner", timeout=5)
    assert out.get("error") == "missing_credentials"
    assert "SDK_DEMO_KEY" in out["missing"]


def test_skill_subprocess_can_import_sdk_and_run_another_skill():
    """End-to-end: a parent skill body uses selfmcp_sdk to invoke a child skill."""
    skills.skill_create(
        "child_greeter",
        "prints a greeting param",
        "import os, json; print('hello', json.loads(os.environ['SELFMCP_PARAMS'])['who'])",
    )
    parent_body = (
        "from selfmcp_sdk import run_skill\n"
        "out = run_skill('child_greeter', params={'who': 'composer'})\n"
        "assert out['exit_code'] == 0, out['stderr']\n"
        "print('parent saw:', out['stdout'].strip())\n"
    )
    parent_id = skills.skill_create("parent_caller", "calls child_greeter", parent_body)["id"]
    res = skills.skill_execute(parent_id, params={}, timeout=20)
    assert res["exit_code"] == 0, res.get("stderr")
    assert "hello composer" in res["stdout"]
    assert "parent saw: hello composer" in res["stdout"]


def test_skill_subprocess_can_search_registry_via_sdk():
    """A skill body can call search_skills() on its own registry."""
    skills.skill_create("weather_fetcher", "Fetch the current weather", "print('w')")
    skills.skill_create("news_fetcher", "Fetch the latest news", "print('n')")
    finder_body = (
        "import json\n"
        "from selfmcp_sdk import search_skills\n"
        "print(json.dumps([h['name'] for h in search_skills('weather')]))\n"
    )
    sid = skills.skill_create("finder", "search via sdk", finder_body)["id"]
    res = skills.skill_execute(sid, params={}, timeout=15)
    assert res["exit_code"] == 0, res.get("stderr")
    names = json.loads(res["stdout"])
    assert "weather_fetcher" in names
