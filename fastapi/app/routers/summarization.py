"""
Summarization Router - Two-stage Summarization API Endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas.summarization import (
    RefineRequest,
    RefineResponse,
    SummarizationRequest,
    SummarizationResponse,
)
from app.services.summarization_service import (
    SummarizationService,
    get_summarization_service,
)


router = APIRouter(prefix="/summarize", tags=["summarization"])


@router.post("", response_model=SummarizationResponse)
async def summarize_text(
    request: SummarizationRequest,
    service: SummarizationService = Depends(get_summarization_service)
) -> SummarizationResponse:
    """
    Tóm tắt văn bản với pipeline hai giai đoạn:
    - Stage 1 (T5-small): Tạo bản tóm tắt thô
    - Stage 2 (BART-base): Tinh chỉnh và hoàn thiện
    
    Lần đầu gọi API sẽ mất thời gian để load models.
    """
    try:
        raw_summary, final_summary = service.summarize(
            text=request.text,
            max_length=request.max_length,
            min_length=request.min_length
        )
        
        return SummarizationResponse(
            raw_summary=raw_summary,
            final_summary=final_summary,
            original_length=len(request.text),
            raw_summary_length=len(raw_summary),
            final_summary_length=len(final_summary)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Summarization failed: {str(e)}"
        )


@router.post("/raw", response_model=dict)
async def summarize_raw_only(
    request: SummarizationRequest,
    service: SummarizationService = Depends(get_summarization_service)
) -> dict:
    """
    Chỉ sử dụng Stage 1 (T5-small) để tạo bản tóm tắt thô.
    Nhanh hơn full pipeline.
    """
    try:
        raw_summary = service.generate_raw_summary(
            text=request.text,
            max_length=request.max_length,
            min_length=request.min_length
        )
        
        return {
            "summary": raw_summary,
            "model": "T5-small",
            "stage": 1,
            "original_length": len(request.text),
            "summary_length": len(raw_summary)
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Summarization failed: {str(e)}"
        )


@router.post("/refine", response_model=RefineResponse)
async def refine_text(
    request: RefineRequest,
    service: SummarizationService = Depends(get_summarization_service)
) -> RefineResponse:
    """
    Chỉ sử dụng Stage 2 (BART-base) để tinh chỉnh văn bản đã có.
    Dùng khi bạn đã có bản tóm tắt thô và muốn polish.
    """
    try:
        refined_text = service.refine_summary(
            text=request.text,
            max_length=request.max_length
        )
        
        improvement_ratio = service.calculate_improvement_ratio(
            request.text,
            refined_text
        )
        
        return RefineResponse(
            original_text=request.text,
            refined_text=refined_text,
            improvement_ratio=improvement_ratio
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Refinement failed: {str(e)}"
        )
