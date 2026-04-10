"""SQLite storage layer for selfMCP.

Single-file database holding:
  - skills            active skill records
  - skill_versions    append-only history (for diff/rollback)
  - skills_fts        FTS5 virtual table over (name, description)
  - skill_embeddings  float32 vector blobs for semantic search
  - skill_credentials optional per-skill secret storage
  - skill_summary_cache  materialized skill_list_summary payload
"""

from __future__ import annotations

import os
import sqlite3
import struct
from contextlib import contextmanager
from typing import Iterator, Sequence

DB_PATH = os.environ.get("SELFMCP_DB_PATH", "selfmcp.db")


SCHEMA = """
CREATE TABLE IF NOT EXISTS skills (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    name              TEXT    NOT NULL UNIQUE,
    description       TEXT    NOT NULL,
    body              TEXT    NOT NULL,
    dependencies_json TEXT    NOT NULL DEFAULT '[]',
    auth_config_json  TEXT,
    version           INTEGER NOT NULL DEFAULT 1,
    is_active         INTEGER NOT NULL DEFAULT 1,
    created_at        REAL    NOT NULL,
    updated_at        REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_skills_active ON skills(is_active);

CREATE TABLE IF NOT EXISTS skill_versions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    skill_id          INTEGER NOT NULL,
    version           INTEGER NOT NULL,
    name              TEXT    NOT NULL,
    description       TEXT    NOT NULL,
    body              TEXT    NOT NULL,
    dependencies_json TEXT    NOT NULL,
    auth_config_json  TEXT,
    changed_at        REAL    NOT NULL,
    FOREIGN KEY (skill_id) REFERENCES skills(id)
);

CREATE INDEX IF NOT EXISTS idx_versions_skill ON skill_versions(skill_id);

-- Regular (not contentless) FTS5 table so we can DELETE/UPDATE rows
-- without going through the special 'delete' command. The storage
-- cost is trivial compared to embeddings.
CREATE VIRTUAL TABLE IF NOT EXISTS skills_fts USING fts5(
    name,
    description
);

CREATE TABLE IF NOT EXISTS skill_embeddings (
    skill_id INTEGER PRIMARY KEY,
    vector   BLOB    NOT NULL,
    dim      INTEGER NOT NULL,
    model    TEXT,
    FOREIGN KEY (skill_id) REFERENCES skills(id)
);

CREATE TABLE IF NOT EXISTS skill_credentials (
    skill_id   INTEGER NOT NULL,
    key        TEXT    NOT NULL,
    value      TEXT    NOT NULL,
    created_at REAL    NOT NULL,
    PRIMARY KEY (skill_id, key),
    FOREIGN KEY (skill_id) REFERENCES skills(id)
);

CREATE TABLE IF NOT EXISTS skill_summary_cache (
    id           INTEGER PRIMARY KEY CHECK (id = 1),
    summary_json TEXT    NOT NULL,
    updated_at   REAL    NOT NULL
);
"""


@contextmanager
def get_conn(db_path: str | None = None) -> Iterator[sqlite3.Connection]:
    """Yield a sqlite3 connection with row access and FK enforcement."""
    conn = sqlite3.connect(db_path or DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: str | None = None) -> None:
    """Create tables if they don't exist."""
    with get_conn(db_path) as conn:
        conn.executescript(SCHEMA)


def pack_vector(vec: Sequence[float]) -> bytes:
    """Pack a list of floats into a float32 blob."""
    return struct.pack(f"{len(vec)}f", *vec)


def unpack_vector(blob: bytes, dim: int) -> list[float]:
    """Unpack a float32 blob back into a list of floats."""
    return list(struct.unpack(f"{dim}f", blob))
