"""Evaluation harness: datasets, QA systems, metrics, and the runner."""

from graphwalk.eval.metrics import Score, normalize, score
from graphwalk.eval.runner import Record, Summary, SystemRun, run_system, sample_questions
from graphwalk.eval.systems import DocIndex, GraphwalkSystem, VectorRAGSystem
from graphwalk.eval.types import EvalQuestion, QASystem, SystemAnswer

__all__ = [
    "DocIndex",
    "EvalQuestion",
    "GraphwalkSystem",
    "QASystem",
    "Record",
    "Score",
    "Summary",
    "SystemAnswer",
    "SystemRun",
    "VectorRAGSystem",
    "normalize",
    "run_system",
    "sample_questions",
    "score",
]
