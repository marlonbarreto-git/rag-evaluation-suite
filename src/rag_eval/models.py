from dataclasses import dataclass, field


@dataclass
class EvalSample:
    """A single RAG evaluation sample containing question, answer, contexts, and optional ground truth."""

    question: str
    answer: str
    contexts: list[str]
    ground_truth: str = ""


@dataclass
class MetricResult:
    """Result of a single metric evaluation, with a normalized score and optional details."""

    name: str
    score: float  # 0.0 to 1.0
    details: dict = field(default_factory=dict)
