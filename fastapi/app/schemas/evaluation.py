"""
Schemas for Evaluation API
Request/Response models cho evaluation endpoints
"""

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# ==================== Request Schemas ====================

class EvaluateSingleRequest(BaseModel):
    """Request body for single evaluation"""
    text: str = Field(..., min_length=10, description="Văn bản gốc cần tóm tắt")
    reference_summary: str = Field(..., min_length=10, description="Bản tóm tắt mẫu (reference)")
    model_name: str = Field(..., description="Tên model để đánh giá (extractive_smart, vit5, bartpho, hybrid)")
    max_length: int = Field(default=150, ge=50, le=500, description="Max length cho summary")
    skip_bert: bool = Field(default=False, description="Skip BERTScore để tính nhanh hơn (~10-12s faster)")


class EvaluateBatchRequest(BaseModel):
    """Request body for batch evaluation"""
    dataset_name: str = Field(default="vietnews", description="Dataset name (vietnews hoặc custom)")
    model_names: List[str] = Field(..., min_items=1, max_items=3, description="Tối đa 3 models để tránh RAM overflow")
    sample_size: int = Field(default=100, ge=10, le=1000, description="Số lượng samples để test")
    max_length: int = Field(default=150, ge=50, le=500, description="Max length cho summary")


class CompareModelsRequest(BaseModel):
    """Request body for model comparison"""
    model_names: List[str] = Field(..., min_items=2, description="Models để so sánh")
    dataset_name: str = Field(default="vietnews", description="Dataset name")
    limit: int = Field(default=10, ge=1, le=100, description="Số lượng evaluations gần nhất")


# ==================== Response Schemas ====================

class EvaluationMetricsResponse(BaseModel):
    """Evaluation metrics response"""
    rouge1: float = Field(..., description="ROUGE-1 F1 score")
    rouge2: float = Field(..., description="ROUGE-2 F1 score")
    rougeL: float = Field(..., description="ROUGE-L F1 score")
    bleu: float = Field(..., description="BLEU score")
    bert_score: Optional[float] = Field(None, description="BERTScore F1 (None nếu skip_bert=True)")
    processing_time_ms: int = Field(..., description="Thời gian xử lý (milliseconds)")


class EvaluateSingleResponse(BaseModel):
    """Response from single evaluation"""
    generated_summary: str
    metrics: EvaluationMetricsResponse
    evaluation_id: str = Field(..., description="ID của evaluation record trong DB")


class OverallMetricsResponse(BaseModel):
    """Overall metrics for batch evaluation"""
    avg_rouge1: float
    avg_rouge2: float
    avg_rougeL: float
    avg_bleu: float
    avg_bert_score: float
    avg_processing_time_ms: float
    total_samples: int


class EvaluateBatchResponse(BaseModel):
    """Response from batch evaluation"""
    evaluation_id: str = Field(..., description="ID để track progress")
    status: Literal["queued", "running", "completed", "failed"]
    message: str = Field(..., description="Status message")


class BatchProgressResponse(BaseModel):
    """Progress update for batch evaluation"""
    evaluation_id: str
    status: Literal["running", "completed", "failed"]
    progress: int = Field(..., ge=0, le=100, description="Progress percentage")
    current_sample: int
    total_samples: int
    overall_metrics: Optional[OverallMetricsResponse] = None
    error_message: Optional[str] = None


class ModelComparisonItem(BaseModel):
    """Comparison metrics for one model"""
    model_name: str
    avg_rouge1: float
    avg_rouge2: float
    avg_rougeL: float
    avg_bleu: float
    avg_bert_score: float
    evaluation_count: int


class CompareModelsResponse(BaseModel):
    """Response from model comparison"""
    comparisons: List[ModelComparisonItem]
    dataset_name: str


class EvaluationHistoryItem(BaseModel):
    """Single evaluation history record"""
    evaluation_id: str
    model_name: str
    dataset_name: str
    created_at: datetime
    overall_metrics: OverallMetricsResponse
    status: Literal["running", "completed", "failed"]


class EvaluationHistoryResponse(BaseModel):
    """Response from evaluation history"""
    evaluations: List[EvaluationHistoryItem]
    total_count: int


class DatasetInfo(BaseModel):
    """Dataset information"""
    name: str
    source: Literal["vietnews", "custom"]
    total_samples: int
    description: str


class DatasetsResponse(BaseModel):
    """Response from datasets listing"""
    datasets: List[DatasetInfo]
