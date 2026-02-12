# rag-evaluation-suite

Evaluation suite for RAG systems. Measures faithfulness, answer relevancy, and context precision using embedding-based metrics. No external LLM calls required.

## Features

- **Answer Relevancy**: Semantic similarity between question and answer
- **Faithfulness**: How well the answer is grounded in retrieved contexts
- **Context Precision**: Whether relevant contexts are ranked higher (Average Precision)
- **Evaluation Runner**: Run all metrics on datasets with aggregate reporting
- **Golden Dataset**: Load evaluation sets from dictionaries for repeatable testing

## Architecture

```
rag_eval/
├── models.py    # EvalSample, MetricResult dataclasses
├── metrics.py   # AnswerRelevancy, Faithfulness, ContextPrecision
└── runner.py    # EvalRunner, EvalReport, SampleReport
```

## Quick Start

```python
from rag_eval.models import EvalSample
from rag_eval.runner import EvalRunner

samples = [
    EvalSample(
        question="What is Python?",
        answer="Python is a programming language used for data science.",
        contexts=["Python is a high-level programming language.", "It is widely used in data science."],
        ground_truth="Python is a high-level programming language.",
    ),
]

runner = EvalRunner()
report = runner.evaluate_dataset(samples)
print(report.summary())
# {'answer_relevancy': 0.85, 'faithfulness': 0.92, 'context_precision': 0.95}
```

## Metrics

| Metric | What it measures | Range |
|--------|-----------------|-------|
| Answer Relevancy | Is the answer relevant to the question? | 0-1 |
| Faithfulness | Is the answer grounded in the provided contexts? | 0-1 |
| Context Precision | Are relevant contexts ranked higher? | 0-1 |

## Development

```bash
uv venv .venv --python 3.12
uv pip install -e ".[dev]"
uv run pytest tests/ -v
```

## Roadmap

- **v2**: Custom metrics (hallucination_rate, citation_accuracy), golden dataset management
- **v3**: CI/CD integration (GitHub Actions), regression detection, trend dashboard

## License

MIT
