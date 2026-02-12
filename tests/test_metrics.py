import pytest

from rag_eval.models import EvalSample, MetricResult
from rag_eval.metrics import AnswerRelevancy, Faithfulness, ContextPrecision


@pytest.fixture(scope="module")
def answer_relevancy():
    return AnswerRelevancy()


@pytest.fixture(scope="module")
def faithfulness():
    return Faithfulness()


@pytest.fixture(scope="module")
def context_precision():
    return ContextPrecision()


# --- AnswerRelevancy ---

class TestAnswerRelevancy:
    def test_high_score_when_relevant(self, answer_relevancy):
        sample = EvalSample(
            question="What is the capital of France?",
            answer="The capital of France is Paris.",
            contexts=["Paris is the capital of France."],
        )
        result = answer_relevancy.evaluate(sample)
        assert result.score > 0.7, f"Expected high relevancy score, got {result.score}"

    def test_low_score_when_irrelevant(self, answer_relevancy):
        sample = EvalSample(
            question="What is the capital of France?",
            answer="Bananas are a great source of potassium.",
            contexts=["Paris is the capital of France."],
        )
        result = answer_relevancy.evaluate(sample)
        assert result.score < 0.5, f"Expected low relevancy score, got {result.score}"

    def test_score_between_zero_and_one(self, answer_relevancy):
        sample = EvalSample(
            question="How does photosynthesis work?",
            answer="Plants convert sunlight into energy through photosynthesis.",
            contexts=["Photosynthesis converts light energy to chemical energy."],
        )
        result = answer_relevancy.evaluate(sample)
        assert 0.0 <= result.score <= 1.0

    def test_returns_metric_result(self, answer_relevancy):
        sample = EvalSample(
            question="What is Python?",
            answer="Python is a programming language.",
            contexts=["Python is a high-level programming language."],
        )
        result = answer_relevancy.evaluate(sample)
        assert isinstance(result, MetricResult)
        assert result.name == "answer_relevancy"

    def test_empty_answer(self, answer_relevancy):
        sample = EvalSample(
            question="What is the capital of France?",
            answer="",
            contexts=["Paris is the capital of France."],
        )
        result = answer_relevancy.evaluate(sample)
        assert 0.0 <= result.score <= 1.0

    def test_empty_question(self, answer_relevancy):
        sample = EvalSample(
            question="",
            answer="The capital of France is Paris.",
            contexts=["Paris is the capital of France."],
        )
        result = answer_relevancy.evaluate(sample)
        assert 0.0 <= result.score <= 1.0


# --- Faithfulness ---

class TestFaithfulness:
    def test_high_score_when_grounded(self, faithfulness):
        sample = EvalSample(
            question="What is the capital of France?",
            answer="The capital of France is Paris. It is located in northern France.",
            contexts=[
                "Paris is the capital and largest city of France.",
                "Paris is located in northern France on the river Seine.",
            ],
        )
        result = faithfulness.evaluate(sample)
        assert result.score > 0.7, f"Expected high faithfulness score, got {result.score}"

    def test_low_score_when_contradicts(self, faithfulness):
        sample = EvalSample(
            question="What is the capital of France?",
            answer="The capital of France is Berlin. It is in Germany.",
            contexts=[
                "Paris is the capital and largest city of France.",
                "Paris is located in northern France on the river Seine.",
            ],
        )
        result = faithfulness.evaluate(sample)
        assert result.score < 0.7, f"Expected lower faithfulness score, got {result.score}"

    def test_empty_contexts_returns_zero(self, faithfulness):
        sample = EvalSample(
            question="What is the capital of France?",
            answer="The capital of France is Paris.",
            contexts=[],
        )
        result = faithfulness.evaluate(sample)
        assert result.score == 0.0

    def test_returns_metric_result(self, faithfulness):
        sample = EvalSample(
            question="What is Python?",
            answer="Python is a programming language.",
            contexts=["Python is a high-level programming language."],
        )
        result = faithfulness.evaluate(sample)
        assert isinstance(result, MetricResult)
        assert result.name == "faithfulness"

    def test_score_between_zero_and_one(self, faithfulness):
        sample = EvalSample(
            question="What is water?",
            answer="Water is H2O. It is essential for life.",
            contexts=["Water is a chemical compound with formula H2O."],
        )
        result = faithfulness.evaluate(sample)
        assert 0.0 <= result.score <= 1.0

    def test_empty_answer(self, faithfulness):
        sample = EvalSample(
            question="What is the capital of France?",
            answer="",
            contexts=["Paris is the capital of France."],
        )
        result = faithfulness.evaluate(sample)
        assert 0.0 <= result.score <= 1.0


# --- ContextPrecision ---

class TestContextPrecision:
    def test_high_score_relevant_context_ranked_first(self, context_precision):
        sample = EvalSample(
            question="What is the capital of France?",
            answer="Paris is the capital of France.",
            contexts=[
                "Paris is the capital and largest city of France.",
                "The Eiffel Tower is a famous landmark.",
                "France is known for its cuisine.",
            ],
            ground_truth="The capital of France is Paris.",
        )
        result = context_precision.evaluate(sample)
        assert result.score > 0.6, f"Expected high precision score, got {result.score}"

    def test_lower_score_relevant_context_ranked_last(self, context_precision):
        sample = EvalSample(
            question="What is the capital of France?",
            answer="Paris is the capital of France.",
            contexts=[
                "The Eiffel Tower is a famous landmark.",
                "France is known for its cuisine.",
                "Paris is the capital and largest city of France.",
            ],
            ground_truth="The capital of France is Paris.",
        )
        result_last = context_precision.evaluate(sample)

        sample_first = EvalSample(
            question="What is the capital of France?",
            answer="Paris is the capital of France.",
            contexts=[
                "Paris is the capital and largest city of France.",
                "The Eiffel Tower is a famous landmark.",
                "France is known for its cuisine.",
            ],
            ground_truth="The capital of France is Paris.",
        )
        result_first = context_precision.evaluate(sample_first)
        assert result_first.score >= result_last.score

    def test_returns_metric_result(self, context_precision):
        sample = EvalSample(
            question="What is Python?",
            answer="Python is a programming language.",
            contexts=["Python is a high-level programming language."],
            ground_truth="Python is a programming language.",
        )
        result = context_precision.evaluate(sample)
        assert isinstance(result, MetricResult)
        assert result.name == "context_precision"

    def test_score_between_zero_and_one(self, context_precision):
        sample = EvalSample(
            question="What is water?",
            answer="Water is H2O.",
            contexts=["Water is a chemical compound.", "Fish live in water."],
            ground_truth="Water is H2O.",
        )
        result = context_precision.evaluate(sample)
        assert 0.0 <= result.score <= 1.0

    def test_empty_contexts(self, context_precision):
        sample = EvalSample(
            question="What is water?",
            answer="Water is H2O.",
            contexts=[],
            ground_truth="Water is H2O.",
        )
        result = context_precision.evaluate(sample)
        assert result.score == 0.0

    def test_no_ground_truth(self, context_precision):
        sample = EvalSample(
            question="What is water?",
            answer="Water is H2O.",
            contexts=["Water is a chemical compound."],
            ground_truth="",
        )
        result = context_precision.evaluate(sample)
        assert 0.0 <= result.score <= 1.0
