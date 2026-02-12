from dataclasses import dataclass

from rag_eval.metrics import AnswerRelevancy, BaseMetric, ContextPrecision, Faithfulness
from rag_eval.models import EvalSample, MetricResult

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
"""Default sentence-transformer model used by all metrics."""


@dataclass
class SampleReport:
    """Evaluation results for a single sample across all metrics."""

    sample: EvalSample
    results: dict[str, MetricResult]


@dataclass
class EvalReport:
    """Aggregated evaluation report over a dataset of samples."""

    sample_reports: list[SampleReport]

    def summary(self) -> dict[str, float]:
        """Return the average score per metric across all samples."""
        if not self.sample_reports:
            return {}
        metric_names = list(self.sample_reports[0].results.keys())
        return {
            name: sum(r.results[name].score for r in self.sample_reports)
            / len(self.sample_reports)
            for name in metric_names
        }


class EvalRunner:
    """Orchestrates metric evaluation over individual samples or full datasets."""

    def __init__(self, metrics: list[BaseMetric] | None = None) -> None:
        if metrics is None:
            self.metrics: list[BaseMetric] = [
                AnswerRelevancy(model_name=DEFAULT_MODEL_NAME),
                Faithfulness(model_name=DEFAULT_MODEL_NAME),
                ContextPrecision(model_name=DEFAULT_MODEL_NAME),
            ]
        else:
            self.metrics = metrics

    def evaluate_sample(self, sample: EvalSample) -> dict[str, MetricResult]:
        """Evaluate a single sample against all configured metrics."""
        return {m.name: m.evaluate(sample) for m in self.metrics}

    def evaluate_dataset(self, samples: list[EvalSample]) -> EvalReport:
        """Evaluate a list of samples and return an aggregated report."""
        reports = [SampleReport(sample=s, results=self.evaluate_sample(s)) for s in samples]
        return EvalReport(sample_reports=reports)

    @staticmethod
    def load_golden_dataset(data: list[dict]) -> list[EvalSample]:
        """Convert a list of raw dictionaries into EvalSample instances."""
        return [EvalSample(**d) for d in data]
