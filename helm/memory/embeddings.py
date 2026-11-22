"""
helm/memory/embeddings.py — Vector embedding generation and cosine similarity.

Uses numpy for fast, offline, deterministic vector math (INSTRUCTIONS.md §1.3).
Provides an n-gram TF-IDF hashing vectorizer for zero-latency offline tool indexing,
with support for pluggable dense embedding providers.
"""

from __future__ import annotations

import math
import re
import zlib
from typing import cast

import numpy as np

# Dimension of local semantic feature space
VECTOR_DIM = 256
EPSILON = 1e-8

# Regex for word tokenization
_WORD_RE = re.compile(r"[a-zA-Z0-9_]+")

# Action and retrieval synonyms for zero-latency semantic mapping
_SYNONYMS: dict[str, str] = {
    "store": "create",
    "save": "create",
    "record": "create",
    "insert": "create",
    "fetch": "read",
    "query": "read",
    "find": "search",
    "show": "read",
}


_MIN_STEM_LEN = 3
_MIN_IES_LEN = 5


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase tokens with subword, stem, and synonym expansion."""
    words = _WORD_RE.findall(text.lower())
    tokens: list[str] = list(words)
    for w in words:
        if "_" in w:
            tokens.extend([part for part in w.split("_") if part])
        # Simple plural/singular normalization
        if w.endswith("ies") and len(w) >= _MIN_IES_LEN:
            tokens.append(w[:-3] + "y")
        elif w.endswith("y") and len(w) >= _MIN_STEM_LEN:
            tokens.append(w[:-1] + "ies")
        elif w.endswith("s") and not w.endswith("ss") and len(w) >= _MIN_STEM_LEN:
            tokens.append(w[:-1])
        else:
            tokens.append(w + "s")
        if w in _SYNONYMS:
            tokens.append(_SYNONYMS[w])
    return tokens


def hash_token_to_index(token: str, dim: int = VECTOR_DIM) -> tuple[int, int]:
    """Hash a token deterministically to an index in [0, dim) and a sign in {-1, 1}."""
    h = zlib.crc32(token.encode("utf-8"))
    idx = h % dim
    sign = 1 if (h // dim) % 2 == 0 else -1
    return idx, sign


def text_to_vector(text: str, dim: int = VECTOR_DIM) -> np.ndarray:
    """
    Convert text to an L2-normalized dense feature vector using signed token hashing.

    Parameters
    ----------
    text:
        Raw input text (tool name, description, schema).
    dim:
        Embedding dimension.

    Returns
    -------
    np.ndarray:
        1D float32 array with unit norm.
    """
    vec = np.zeros(dim, dtype=np.float32)
    tokens = tokenize(text)
    if not tokens:
        return vec

    # Compute term frequencies with logarithmic dampening
    tf: dict[str, int] = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1

    for token, count in tf.items():
        idx, sign = hash_token_to_index(token, dim)
        weight = (1.0 + math.log(count)) * sign
        vec[idx] += weight

    # L2 normalize
    norm = float(np.linalg.norm(vec))
    if norm > EPSILON:
        vec /= norm

    return vec


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two 1D vectors."""
    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    if norm_a < EPSILON or norm_b < EPSILON:
        return 0.0
    similarity = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
    return max(0.0, min(1.0, similarity))


def batch_cosine_similarity(query_vec: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarities between a 1D query vector and a 2D matrix of document vectors.

    Parameters
    ----------
    query_vec:
        1D array of shape (dim,).
    doc_matrix:
        2D array of shape (n_docs, dim).

    Returns
    -------
    np.ndarray:
        1D array of similarity scores in [0.0, 1.0].
    """
    if doc_matrix.size == 0:
        return np.empty(0, dtype=np.float32)

    query_norm = float(np.linalg.norm(query_vec))
    if query_norm < EPSILON:
        return np.zeros(doc_matrix.shape[0], dtype=np.float32)

    # Assumes doc_matrix rows are normalized or normalizes on the fly
    doc_norms = np.linalg.norm(doc_matrix, axis=1)
    doc_norms = np.where(doc_norms < EPSILON, 1.0, doc_norms)

    dots = np.dot(doc_matrix, query_vec)
    scores = dots / (doc_norms * query_norm)
    return cast("np.ndarray", np.clip(scores, 0.0, 1.0))
