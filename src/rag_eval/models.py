from dataclasses import dataclass, field


@dataclass
class EvalSample:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str = ""


@dataclass
class MetricResult:
    name: str
    score: float  # 0.0 to 1.0
    details: dict = field(default_factory=dict)
