# RAG Evaluation Suite

[![CI](https://github.com/marlonbarreto-git/rag-evaluation-suite/actions/workflows/ci.yml/badge.svg)](https://github.com/marlonbarreto-git/rag-evaluation-suite/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Evaluation framework for RAG systems measuring faithfulness, answer relevancy, and context precision.

## Overview

RAG Evaluation Suite provides embedding-based metrics to assess RAG pipeline quality without requiring an LLM judge. It evaluates three dimensions: whether the answer is relevant to the question (Answer Relevancy), whether the answer is grounded in the provided context (Faithfulness), and whether the retrieved contexts are relevant to the expected answer (Context Precision via Average Precision).

## Architecture

```
EvalSample (question, answer, contexts, ground_truth)
  |
  v
EvalRunner
  |
  +---> AnswerRelevancy
  |        cosine_similarity(question_emb, answer_emb)
  |
  +---> Faithfulness
  |        per-sentence max similarity against contexts
  |        score = mean(sentence_scores)
  |
  +---> ContextPrecision
  |        Average Precision of context relevance
  |        threshold = 0.5 cosine similarity
  |
  v
EvalReport
  |
  +---> Per-sample MetricResult (name, score 0-1, details)
  +---> Summary (average score per metric across dataset)
```

## Features

- Answer Relevancy: cosine similarity between question and answer embeddings
- Faithfulness: per-sentence grounding check against retrieved contexts
- Context Precision: Average Precision measuring context ordering quality
- No LLM judge required -- purely embedding-based evaluation
- Batch evaluation with dataset-level summary statistics
- Golden dataset loading from dictionaries

## Tech Stack

- Python 3.11+
- sentence-transformers
- NumPy
- Pydantic

## Quick Start

```bash
git clone https://github.com/marlonbarreto-git/rag-evaluation-suite.git
cd rag-evaluation-suite
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Project Structure

```
src/rag_eval/
  __init__.py
  models.py    # EvalSample and MetricResult dataclasses
  metrics.py   # BaseMetric ABC + AnswerRelevancy, Faithfulness, ContextPrecision
  runner.py    # EvalRunner with batch evaluation and reporting
tests/
  test_metrics.py
  test_runner.py
```

## Testing

```bash
pytest -v --cov=src/rag_eval
```

29 tests covering all three metrics, edge cases, batch evaluation, and report aggregation.

## License

MIT