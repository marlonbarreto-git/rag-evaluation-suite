from abc import ABC, abstractmethod

import numpy as np
from sentence_transformers import SentenceTransformer

from rag_eval.models import EvalSample, MetricResult

CONTEXT_RELEVANCE_THRESHOLD = 0.5
"""Minimum cosine similarity for a context chunk to be considered relevant."""


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences by common delimiters."""
    if not text.strip():
        return []
    separators = [". ", "? ", "! "]
    sentences = [text]
    for sep in separators:
        new_sentences = []
        for s in sentences:
            new_sentences.extend(s.split(sep))
        sentences = new_sentences
    return [s.strip() for s in sentences if s.strip()]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors, returning 0.0 for zero-norm inputs."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class BaseMetric(ABC):
    """Abstract base class for all RAG evaluation metrics."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model = SentenceTransformer(model_name)

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique name identifier for this metric."""
        ...

    @abstractmethod
    def evaluate(self, sample: EvalSample) -> MetricResult:
        """Evaluate a single sample and return a MetricResult."""
        ...


class AnswerRelevancy(BaseMetric):
    """Measures how relevant the answer is to the question using embedding similarity."""

    @property
    def name(self) -> str:
        return "answer_relevancy"

    def evaluate(self, sample: EvalSample) -> MetricResult:
        if not sample.question.strip() or not sample.answer.strip():
            return MetricResult(name=self.name, score=0.0, details={"reason": "empty input"})

        q_emb = self._model.encode(sample.question)
        a_emb = self._model.encode(sample.answer)
        score = _cosine_similarity(q_emb, a_emb)
        score = max(0.0, min(1.0, score))
        return MetricResult(name=self.name, score=score)


class Faithfulness(BaseMetric):
    """Measures how grounded the answer is in the provided contexts."""

    @property
    def name(self) -> str:
        return "faithfulness"

    def evaluate(self, sample: EvalSample) -> MetricResult:
        if not sample.contexts:
            return MetricResult(name=self.name, score=0.0, details={"reason": "no contexts"})

        sentences = _split_sentences(sample.answer)
        if not sentences:
            return MetricResult(name=self.name, score=0.0, details={"reason": "empty answer"})

        context_embeddings = self._model.encode(sample.contexts)
        sentence_scores = []
        for sentence in sentences:
            s_emb = self._model.encode(sentence)
            max_sim = max(
                _cosine_similarity(s_emb, c_emb) for c_emb in context_embeddings
            )
            sentence_scores.append(max(0.0, max_sim))

        score = float(np.mean(sentence_scores))
        score = max(0.0, min(1.0, score))
        return MetricResult(
            name=self.name,
            score=score,
            details={"per_sentence_scores": sentence_scores},
        )


class ContextPrecision(BaseMetric):
    """Measures the precision of retrieved contexts using Average Precision."""

    @property
    def name(self) -> str:
        return "context_precision"

    def evaluate(self, sample: EvalSample) -> MetricResult:
        if not sample.contexts:
            return MetricResult(name=self.name, score=0.0, details={"reason": "no contexts"})

        if sample.ground_truth.strip():
            reference = sample.ground_truth
        else:
            reference = sample.answer

        ref_emb = self._model.encode(reference)
        context_embeddings = self._model.encode(sample.contexts)

        similarities = [
            _cosine_similarity(ref_emb, c_emb) for c_emb in context_embeddings
        ]

        relevance = [1 if sim >= CONTEXT_RELEVANCE_THRESHOLD else 0 for sim in similarities]

        num_relevant = sum(relevance)
        if num_relevant == 0:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"similarities": similarities, "relevance": relevance},
            )

        # Average Precision: AP = sum(precision@k * rel(k)) / num_relevant
        cumulative_relevant = 0
        ap_sum = 0.0
        for k, rel in enumerate(relevance):
            if rel == 1:
                cumulative_relevant += 1
                precision_at_k = cumulative_relevant / (k + 1)
                ap_sum += precision_at_k

        score = ap_sum / num_relevant
        score = max(0.0, min(1.0, score))
        return MetricResult(
            name=self.name,
            score=score,
            details={"similarities": similarities, "relevance": relevance},
        )
