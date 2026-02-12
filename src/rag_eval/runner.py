from dataclasses import dataclass, field

from rag_eval.metrics import AnswerRelevancy, BaseMetric, ContextPrecision, Faithfulness
from rag_eval.models import EvalSample, MetricResult


@dataclass
class SampleReport:
    sample: EvalSample
    results: dict[str, MetricResult]


@dataclass
class EvalReport:
    sample_reports: list[SampleReport]

    def summary(self) -> dict[str, float]:
        """Average score per metric across all samples."""
        if not self.sample_reports:
            return {}
        metric_names = list(self.sample_reports[0].results.keys())
        return {
            name: sum(r.results[name].score for r in self.sample_reports)
            / len(self.sample_reports)
            for name in metric_names
        }


class EvalRunner:
    def __init__(self, metrics: list[BaseMetric] | None = None):
        if metrics is None:
            model_name = "all-MiniLM-L6-v2"
            self.metrics: list[BaseMetric] = [
                AnswerRelevancy(model_name=model_name),
                Faithfulness(model_name=model_name),
                ContextPrecision(model_name=model_name),
            ]
        else:
            self.metrics = metrics

    def evaluate_sample(self, sample: EvalSample) -> dict[str, MetricResult]:
        return {m.name: m.evaluate(sample) for m in self.metrics}

    def evaluate_dataset(self, samples: list[EvalSample]) -> EvalReport:
        reports = [SampleReport(sample=s, results=self.evaluate_sample(s)) for s in samples]
        return EvalReport(sample_reports=reports)

    @staticmethod
    def load_golden_dataset(data: list[dict]) -> list[EvalSample]:
        return [EvalSample(**d) for d in data]
