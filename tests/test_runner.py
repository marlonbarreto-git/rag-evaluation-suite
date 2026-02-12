from unittest.mock import MagicMock

import pytest

from rag_eval.models import EvalSample, MetricResult
from rag_eval.runner import EvalReport, EvalRunner, SampleReport


def _make_sample(**overrides):
    defaults = {
        "question": "What is RAG?",
        "answer": "Retrieval-Augmented Generation",
        "contexts": ["RAG combines retrieval with generation."],
        "ground_truth": "RAG is Retrieval-Augmented Generation.",
    }
    defaults.update(overrides)
    return EvalSample(**defaults)


def _make_fake_metric(name: str, score: float):
    metric = MagicMock()
    metric.name = name
    metric.evaluate.return_value = MetricResult(name=name, score=score)
    return metric


# --- EvalRunner init ---


class TestEvalRunnerInit:
    def test_init_with_default_metrics(self):
        runner = EvalRunner()
        assert len(runner.metrics) == 3
        names = {m.name for m in runner.metrics}
        assert names == {"answer_relevancy", "faithfulness", "context_precision"}

    def test_init_with_custom_metrics(self):
        m1 = _make_fake_metric("custom_metric", 0.5)
        runner = EvalRunner(metrics=[m1])
        assert len(runner.metrics) == 1
        assert runner.metrics[0].name == "custom_metric"


# --- evaluate_sample ---


class TestEvaluateSample:
    def test_returns_dict_of_metric_results(self):
        m1 = _make_fake_metric("relevancy", 0.9)
        m2 = _make_fake_metric("faithfulness", 0.8)
        runner = EvalRunner(metrics=[m1, m2])
        sample = _make_sample()

        result = runner.evaluate_sample(sample)

        assert isinstance(result, dict)
        assert set(result.keys()) == {"relevancy", "faithfulness"}
        assert result["relevancy"].score == 0.9
        assert result["faithfulness"].score == 0.8

    def test_calls_evaluate_on_each_metric(self):
        m1 = _make_fake_metric("a", 0.5)
        m2 = _make_fake_metric("b", 0.6)
        runner = EvalRunner(metrics=[m1, m2])
        sample = _make_sample()

        runner.evaluate_sample(sample)

        m1.evaluate.assert_called_once_with(sample)
        m2.evaluate.assert_called_once_with(sample)


# --- evaluate_dataset ---


class TestEvaluateDataset:
    def test_returns_eval_report(self):
        m1 = _make_fake_metric("m1", 0.7)
        runner = EvalRunner(metrics=[m1])
        samples = [_make_sample(), _make_sample(question="Q2")]

        report = runner.evaluate_dataset(samples)

        assert isinstance(report, EvalReport)
        assert len(report.sample_reports) == 2
        for sr in report.sample_reports:
            assert isinstance(sr, SampleReport)
            assert "m1" in sr.results

    def test_per_sample_results(self):
        scores = iter([0.8, 0.6])
        metric = MagicMock()
        metric.name = "score"
        metric.evaluate.side_effect = lambda s: MetricResult(name="score", score=next(scores))

        runner = EvalRunner(metrics=[metric])
        samples = [_make_sample(question="Q1"), _make_sample(question="Q2")]

        report = runner.evaluate_dataset(samples)

        assert report.sample_reports[0].results["score"].score == 0.8
        assert report.sample_reports[1].results["score"].score == 0.6

    def test_empty_dataset(self):
        m1 = _make_fake_metric("m1", 0.5)
        runner = EvalRunner(metrics=[m1])

        report = runner.evaluate_dataset([])

        assert isinstance(report, EvalReport)
        assert report.sample_reports == []


# --- EvalReport.summary ---


class TestEvalReportSummary:
    def test_summary_returns_averages_per_metric(self):
        s1 = SampleReport(
            sample=_make_sample(),
            results={
                "rel": MetricResult(name="rel", score=0.8),
                "faith": MetricResult(name="faith", score=0.6),
            },
        )
        s2 = SampleReport(
            sample=_make_sample(),
            results={
                "rel": MetricResult(name="rel", score=0.4),
                "faith": MetricResult(name="faith", score=1.0),
            },
        )
        report = EvalReport(sample_reports=[s1, s2])

        summary = report.summary()

        assert summary == pytest.approx({"rel": 0.6, "faith": 0.8})

    def test_summary_empty_reports(self):
        report = EvalReport(sample_reports=[])
        assert report.summary() == {}


# --- load_golden_dataset ---


class TestLoadGoldenDataset:
    def test_load_from_list_of_dicts(self):
        data = [
            {
                "question": "Q1",
                "answer": "A1",
                "contexts": ["c1"],
                "ground_truth": "GT1",
            },
            {
                "question": "Q2",
                "answer": "A2",
                "contexts": ["c2", "c3"],
                "ground_truth": "GT2",
            },
        ]

        samples = EvalRunner.load_golden_dataset(data)

        assert len(samples) == 2
        assert all(isinstance(s, EvalSample) for s in samples)
        assert samples[0].question == "Q1"
        assert samples[1].contexts == ["c2", "c3"]

    def test_load_with_missing_ground_truth_uses_default(self):
        data = [{"question": "Q", "answer": "A", "contexts": ["c"]}]

        samples = EvalRunner.load_golden_dataset(data)

        assert samples[0].ground_truth == ""
