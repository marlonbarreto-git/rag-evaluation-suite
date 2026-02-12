import pytest

from rag_eval.models import EvalSample, MetricResult


# --- EvalSample ---


class TestEvalSample:
    def test_instantiation_with_all_fields(self):
        sample = EvalSample(
            question="What is RAG?",
            answer="Retrieval-Augmented Generation",
            contexts=["RAG combines retrieval with generation."],
            ground_truth="RAG is Retrieval-Augmented Generation.",
        )
        assert sample.question == "What is RAG?"
        assert sample.answer == "Retrieval-Augmented Generation"
        assert sample.contexts == ["RAG combines retrieval with generation."]
        assert sample.ground_truth == "RAG is Retrieval-Augmented Generation."

    def test_ground_truth_defaults_to_empty_string(self):
        sample = EvalSample(
            question="Q",
            answer="A",
            contexts=["c"],
        )
        assert sample.ground_truth == ""

    def test_contexts_can_be_empty_list(self):
        sample = EvalSample(
            question="Q",
            answer="A",
            contexts=[],
        )
        assert sample.contexts == []

    def test_contexts_multiple_items(self):
        contexts = ["first chunk", "second chunk", "third chunk"]
        sample = EvalSample(
            question="Q",
            answer="A",
            contexts=contexts,
        )
        assert len(sample.contexts) == 3
        assert sample.contexts == contexts

    def test_empty_strings_allowed(self):
        sample = EvalSample(
            question="",
            answer="",
            contexts=[""],
            ground_truth="",
        )
        assert sample.question == ""
        assert sample.answer == ""
        assert sample.contexts == [""]
        assert sample.ground_truth == ""

    def test_missing_required_field_raises_error(self):
        with pytest.raises(TypeError):
            EvalSample(question="Q", answer="A")  # missing contexts

    def test_missing_question_raises_error(self):
        with pytest.raises(TypeError):
            EvalSample(answer="A", contexts=["c"])  # missing question

    def test_missing_answer_raises_error(self):
        with pytest.raises(TypeError):
            EvalSample(question="Q", contexts=["c"])  # missing answer

    def test_equality(self):
        s1 = EvalSample(question="Q", answer="A", contexts=["c"], ground_truth="GT")
        s2 = EvalSample(question="Q", answer="A", contexts=["c"], ground_truth="GT")
        assert s1 == s2

    def test_inequality_different_fields(self):
        s1 = EvalSample(question="Q1", answer="A", contexts=["c"])
        s2 = EvalSample(question="Q2", answer="A", contexts=["c"])
        assert s1 != s2


# --- MetricResult ---


class TestMetricResult:
    def test_instantiation_with_all_fields(self):
        result = MetricResult(
            name="test_metric",
            score=0.85,
            details={"reason": "high similarity"},
        )
        assert result.name == "test_metric"
        assert result.score == 0.85
        assert result.details == {"reason": "high similarity"}

    def test_details_defaults_to_empty_dict(self):
        result = MetricResult(name="m", score=0.5)
        assert result.details == {}

    def test_score_zero(self):
        result = MetricResult(name="m", score=0.0)
        assert result.score == 0.0

    def test_score_one(self):
        result = MetricResult(name="m", score=1.0)
        assert result.score == 1.0

    def test_details_with_nested_data(self):
        details = {"per_sentence_scores": [0.9, 0.8, 0.7], "count": 3}
        result = MetricResult(name="m", score=0.8, details=details)
        assert result.details["per_sentence_scores"] == [0.9, 0.8, 0.7]
        assert result.details["count"] == 3

    def test_missing_required_name_raises_error(self):
        with pytest.raises(TypeError):
            MetricResult(score=0.5)  # missing name

    def test_missing_required_score_raises_error(self):
        with pytest.raises(TypeError):
            MetricResult(name="m")  # missing score

    def test_equality(self):
        r1 = MetricResult(name="m", score=0.5, details={"k": "v"})
        r2 = MetricResult(name="m", score=0.5, details={"k": "v"})
        assert r1 == r2

    def test_inequality_different_score(self):
        r1 = MetricResult(name="m", score=0.5)
        r2 = MetricResult(name="m", score=0.6)
        assert r1 != r2

    def test_details_default_not_shared_between_instances(self):
        r1 = MetricResult(name="m1", score=0.5)
        r2 = MetricResult(name="m2", score=0.6)
        r1.details["key"] = "value"
        assert "key" not in r2.details
