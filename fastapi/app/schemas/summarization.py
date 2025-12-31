"""
Schemas for Summarization API
Two-stage summarization: T5-small (draft) -> BART-base (polish)
"""

from typing import Optional

from pydantic import BaseModel, Field


class SummarizationRequest(BaseModel):
    """Request body for summarization endpoint"""
    
    text: str = Field(
        ...,
        min_length=10,
        description="Văn bản cần tóm tắt (tối thiểu 10 ký tự)"
    )
    max_length: int = Field(
        default=150,
        ge=50,
        le=500,
        description="Độ dài tối đa của bản tóm tắt"
    )
    min_length: int = Field(
        default=30,
        ge=10,
        le=200,
        description="Độ dài tối thiểu của bản tóm tắt"
    )


class SummarizationResponse(BaseModel):
    """Response from summarization endpoint"""
    
    raw_summary: str = Field(
        ...,
        description="Bản tóm tắt thô từ T5-small (Stage 1)"
    )
    final_summary: str = Field(
        ...,
        description="Bản tóm tắt hoàn chỉnh từ BART-base (Stage 2)"
    )
    original_length: int = Field(
        ...,
        description="Độ dài văn bản gốc (số ký tự)"
    )
    raw_summary_length: int = Field(
        ...,
        description="Độ dài bản tóm tắt thô"
    )
    final_summary_length: int = Field(
        ...,
        description="Độ dài bản tóm tắt cuối"
    )


class RefineRequest(BaseModel):
    """Request body for refine-only endpoint"""
    
    text: str = Field(
        ...,
        min_length=10,
        description="Văn bản thô cần tinh chỉnh"
    )
    max_length: int = Field(
        default=150,
        ge=50,
        le=500,
        description="Độ dài tối đa sau khi tinh chỉnh"
    )


class RefineResponse(BaseModel):
    """Response from refine endpoint"""
    
    original_text: str
    refined_text: str
    improvement_ratio: float = Field(
        ...,
        description="Tỷ lệ thay đổi so với bản gốc (0-1)"
    )
