"""
Summarization Router - BART-large-cnn with Pre/Post Processing
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
    Tóm tắt văn bản với BART-large-cnn + pre/post processing:
    - Pre-processing: Clean text, extract entities
    - Summarization: BART-large-cnn
    - Post-processing: Remove redundancy
    
    Lần đầu gọi API sẽ mất thời gian để load model (~1.6GB).
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


@router.post("/balanced", response_model=dict)
async def summarize_balanced(
    request: SummarizationRequest,
    service: SummarizationService = Depends(get_summarization_service)
) -> dict:
    """
    Tóm tắt cân bằng theo topic (Topic-Balanced Summarization):
    - Chia văn bản thành các topic riêng biệt
    - Tóm tắt mỗi topic với quota độ dài bằng nhau
    - Kết hợp thành bản tóm tắt cuối cùng
    
    Đảm bảo mỗi topic được đề cập đầy đủ - giải quyết vấn đề
    BART ưu tiên nội dung ở đầu văn bản.
    """
    try:
        raw_combined, final_summary, topic_summaries = service.summarize_balanced(
            text=request.text,
            max_length=request.max_length,
            min_length=request.min_length
        )
        
        return {
            "raw_summary": raw_combined,
            "final_summary": final_summary,
            "original_length": len(request.text),
            "final_summary_length": len(final_summary),
            "topics_found": len(topic_summaries),
            "topic_summaries": topic_summaries
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Balanced summarization failed: {str(e)}"
        )


@router.post("/detailed", response_model=dict)
async def summarize_with_details(
    request: SummarizationRequest,
    service: SummarizationService = Depends(get_summarization_service)
) -> dict:
    """
    Tóm tắt văn bản với phân tích chi tiết:
    - raw_summary: Bản tóm tắt gốc từ BART
    - final_summary: Bản tóm tắt sau post-processing
    - entities: Danh sách entities quan trọng
    - entity_coverage: Entities nào có mặt trong summary
    - topics: Các topic được xác định
    - topic_balance: Độ bao phủ mỗi topic
    
    Endpoint này hữu ích để debug và đánh giá chất lượng.
    """
    try:
        result = service.summarize_with_details(
            text=request.text,
            max_length=request.max_length,
            min_length=request.min_length
        )
        return result
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
    Chỉ tạo bản tóm tắt thô (không có post-processing).
    Dùng để so sánh với bản đã post-process.
    """
    try:
        raw_summary = service.generate_raw_summary(
            text=request.text,
            max_length=request.max_length,
            min_length=request.min_length
        )
        
        return {
            "summary": raw_summary,
            "model": "facebook/bart-large-cnn",
            "post_processed": False,
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
    Chỉ chạy post-processing trên văn bản đã có.
    Dùng để test hiệu quả của post-processing.
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

