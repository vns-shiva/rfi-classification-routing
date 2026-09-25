"""Description-similarity signal, vendored verbatim from
../meeting-action-extraction/code/embeddings.py (domain-agnostic: it only
ever sees pairs of free-text strings).

Two interchangeable implementations of the same `.pairwise(a_texts, b_texts)
-> np.ndarray` contract:

- TfidfSimilarity: scikit-learn TF-IDF + cosine similarity. Zero-dependency
  placeholder, kept as a fast fallback and as the fixture the matching.py
  smoke test was originally tuned against.
- BGESimilarity: the paper's actual specified signal ([10], BAAI/bge-large-en-v1.5
  via sentence-transformers), installed in the C:\\pyenv virtualenv on this
  machine (torch 2.14.0+cpu, sentence-transformers 6.1.0). Requires running
  under that interpreter — the system Python has neither torch nor
  sentence-transformers installed.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfSimilarity:
    """TF-IDF + cosine-similarity placeholder for BGE-large-en-v1.5. Fits a
    single vectorizer over the union of both item sets' descriptions so
    similarity is comparable across the pair."""

    name = "tfidf"

    def pairwise(self, a_texts: list[str], b_texts: list[str]) -> np.ndarray:
        n, m = len(a_texts), len(b_texts)
        if n == 0 or m == 0:
            return np.zeros((n, m))

        # Exact-text pairs are similarity 1.0 regardless of what the
        # vectorizer below does with them — this matters when the whole
        # combined corpus collapses to one unique string (e.g. two identical
        # extractions of the same item from overlapping chunks), where a
        # single-document TF-IDF fit is otherwise meaningless.
        exact = np.array([[a == b for b in b_texts] for a in a_texts], dtype=bool)

        sim = np.zeros((n, m))
        try:
            vec = TfidfVectorizer(stop_words="english")
            matrix = vec.fit_transform(a_texts + b_texts)
            a_vecs = matrix[:n]
            b_vecs = matrix[n:]
            sim = cosine_similarity(a_vecs, b_vecs)
        except ValueError:
            # Empty vocabulary (corpus is entirely stop words) — fall back to
            # exact-match-only scoring below rather than crashing the run.
            pass

        return np.where(exact, 1.0, sim)


class BGESimilarity:
    """Real IV.A/IV.D signal: BAAI/bge-large-en-v1.5 sentence embeddings,
    L2-normalized, scored by cosine similarity (== dot product once
    normalized). Model is loaded once and cached on the instance."""

    name = "bge"
    MODEL_NAME = "BAAI/bge-large-en-v1.5"

    def __init__(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed in this interpreter. "
                "Run with C:\\pyenv\\Scripts\\python.exe, where it and torch "
                "are installed, or `pip install sentence-transformers`."
            ) from exc
        self._model = SentenceTransformer(self.MODEL_NAME)

    def pairwise(self, a_texts: list[str], b_texts: list[str]) -> np.ndarray:
        if not a_texts or not b_texts:
            return np.zeros((len(a_texts), len(b_texts)))
        a_emb = self._model.encode(a_texts, normalize_embeddings=True)
        b_emb = self._model.encode(b_texts, normalize_embeddings=True)
        return np.asarray(a_emb) @ np.asarray(b_emb).T


# Backward-compatible name: matching.py's original stub was called
# EmbeddingSimilarity and raised NotImplementedError. Now that BGE is
# installed, that name is the real thing.
EmbeddingSimilarity = BGESimilarity


def build_similarity(name: str):
    if name == "tfidf":
        return TfidfSimilarity()
    if name == "bge":
        return BGESimilarity()
    raise ValueError(f"Unknown similarity model: {name!r} (expected 'tfidf' or 'bge')")
