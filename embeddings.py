"""Embedding generation and similarity.

Uses LiteLLM when an API key is configured (defaulting to
OpenAI's ``text-embedding-3-small``). Falls back to a deterministic
local hash-based embedding so the server runs offline and in tests.

The local fallback is not semantically rich, but it keeps the
same interface — swap it for a real model by setting
``OPENAI_API_KEY`` (or any LiteLLM-supported provider env var) and,
optionally, ``SELFMCP_EMBED_MODEL``.
"""

from __future__ import annotations

import hashlib
import math
import os
from typing import Sequence

LOCAL_EMBED_DIM = 256


def _local_embedding(text: str, dim: int = LOCAL_EMBED_DIM) -> list[float]:
    """Deterministic hash-based embedding for offline use.

    Tokenizes on whitespace, hashes each token into a handful of
    sign-weighted buckets, then L2-normalizes. Good enough for
    keyword-ish matching and for unit tests.
    """
    vec = [0.0] * dim
    tokens = [t for t in text.lower().split() if t]
    for tok in tokens:
        h = hashlib.sha256(tok.encode("utf-8")).digest()
        # Hash into 4 buckets per token; half with positive sign, half negative.
        for i in range(4):
            idx = int.from_bytes(h[i * 4 : i * 4 + 4], "big") % dim
            sign = 1.0 if (h[16 + i] & 1) else -1.0
            vec[idx] += sign
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


def get_embedding(text: str) -> tuple[list[float], str]:
    """Return ``(vector, model_name)`` for ``text``.

    Tries LiteLLM first; on any failure or missing credentials
    falls back to the local hash embedder. The model name is
    recorded so we can detect dimension/model mismatches later.
    """
    text = (text or "").strip() or " "
    use_remote = bool(
        os.environ.get("SELFMCP_USE_LITELLM")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
    )
    if use_remote:
        try:
            import litellm  # type: ignore

            model = os.environ.get("SELFMCP_EMBED_MODEL", "text-embedding-3-small")
            resp = litellm.embedding(model=model, input=[text])
            vec = list(resp["data"][0]["embedding"])
            return vec, model
        except Exception:
            # Fall through to local embedder rather than failing the write.
            pass
    return _local_embedding(text), "local-hash-v1"


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity in pure Python. Returns 0 for mismatched shapes."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0 or nb == 0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))
