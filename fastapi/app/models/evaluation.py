"""
Evaluation Models for MongoDB
Định nghĩa schema cho evaluation logs và results
"""

from datetime import datetime
from typing import List, Literal, Optional, TypedDict


class EvaluationMetrics(TypedDict):
    """Metrics for a single evaluation"""
    rouge1: float
    rouge2: float
    rougeL: float
    bleu: float
    bert_score: float
    processing_time_ms: int


class SummaryDataItem(TypedDict):
    """Individual summary evaluation result"""
    input_text: str
    reference_summary: str
    generated_summary: str
    metrics: EvaluationMetrics


class OverallMetrics(TypedDict):
    """Aggregated metrics for an evaluation run"""
    avg_rouge1: float
    avg_rouge2: float
    avg_rougeL: float
    avg_bleu: float
    avg_bert_score: float
    avg_processing_time_ms: float
    total_samples: int


class EvaluationDocument(TypedDict, total=False):
    """
    MongoDB document for evaluation results
    Collection: evaluation_logs
    """
    _id: str
    user_id: str
    model_name: str
    dataset_name: str
    created_at: datetime
    summary_data: List[SummaryDataItem]
    overall_metrics: OverallMetrics
    status: Literal["running", "completed", "failed"]
    error_message: Optional[str]


class DatasetDocument(TypedDict, total=False):
    """
    MongoDB document for dataset metadata
    Collection: datasets
    """
    _id: str
    name: str
    source: Literal["vietnews", "custom"]
    total_samples: int
    description: str
    created_at: datetime
    file_path: Optional[str]  # For custom datasets
