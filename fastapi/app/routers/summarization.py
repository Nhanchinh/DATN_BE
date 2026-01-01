"""
Summarization Router - BART-large-cnn with Pre/Post Processing + mT5 Multilingual
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
from app.services.multilingual_service import (
    MultilingualSummarizationService,
    get_multilingual_service,
)
from app.services.extractive_service import (
    ExtractiveSummarizationService,
    get_extractive_service,
)
from app.services.hybrid_service import (
    HybridSummarizationService,
    get_hybrid_service,
)
from app.services.bartpho_service import (
    BARTphoService,
    get_bartpho_service,
)
from app.services.qwen_service import (
    QwenService,
    get_qwen_service,
)
from app.services.qwen_15b_service import (
    Qwen15BService,
    get_qwen_15b_service,
)
from app.services.qwen_strict_service import (
    QwenStrictService,
    get_qwen_strict_service,
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


@router.post("/multilingual", response_model=dict)
async def summarize_multilingual(
    request: SummarizationRequest,
    service: MultilingualSummarizationService = Depends(get_multilingual_service)
) -> dict:
    """
    Tóm tắt tiếng Việt với ViT5 (VietAI).
    
    Model được VietAI fine-tune đặc biệt cho tóm tắt tin tức tiếng Việt.
    
    Lần đầu gọi sẽ download model (~900MB).
    """
    try:
        raw_summary, processed_summary = service.summarize(
            text=request.text,
            max_length=request.max_length,
            min_length=request.min_length
        )
        
        return {
            "raw_summary": raw_summary,
            "final_summary": processed_summary,
            "model": "VietAI/vit5-base-vietnews-summarization",
            "original_length": len(request.text),
            "final_summary_length": len(processed_summary),
            "language": "vi"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vietnamese summarization failed: {str(e)}"
        )


@router.get("/multilingual/info", response_model=dict)
async def get_multilingual_info(
    service: MultilingualSummarizationService = Depends(get_multilingual_service)
) -> dict:
    """Lấy thông tin về model đa ngôn ngữ"""
    return service.get_model_info()


@router.post("/extractive", response_model=dict)
async def summarize_extractive(
    request: SummarizationRequest,
    service: ExtractiveSummarizationService = Depends(get_extractive_service)
) -> dict:
    """
    Tóm tắt trích xuất với PhoBERT (Extractive Summarization).
    
    **AN TOÀN 100%** - Không bao giờ hallucinate!
    
    Chỉ trích xuất các câu quan trọng nhất từ văn bản gốc,
    không viết lại hay thêm bất kỳ thông tin nào.
    
    Phù hợp cho: Mô tả sách, tài liệu kỹ thuật, nội dung cần chính xác.
    
    Lần đầu gọi sẽ download model (~400MB).
    """
    try:
        # Calculate ratio based on max_length vs original length
        ratio = min(0.5, max(0.2, request.max_length / len(request.text)))
        
        summary, extracted_sentences = service.summarize(
            text=request.text,
            ratio=ratio
        )
        
        return {
            "summary": summary,
            "extracted_sentences": extracted_sentences,
            "num_sentences_extracted": len(extracted_sentences),
            "model": "vinai/phobert-base",
            "method": "extractive",
            "hallucination_risk": "ZERO",
            "original_length": len(request.text),
            "summary_length": len(summary),
            "language": "vi"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Extractive summarization failed: {str(e)}"
        )


@router.post("/chunked", response_model=dict)
async def summarize_chunked(
    request: SummarizationRequest,
    service: ExtractiveSummarizationService = Depends(get_extractive_service)
) -> dict:
    """
    Tóm tắt trích xuất theo chunks (Chunk-based Extractive).
    
    **Tốt nhất cho tiếng Việt - An toàn 100%!**
    
    Pipeline:
    1. Chia văn bản thành nhiều chunks (nhóm 3 câu)
    2. Trích xuất 1 câu quan trọng nhất từ mỗi chunk
    3. Ghép các câu lại thành summary hoàn chỉnh
    
    **Ưu điểm:**
    - Không bao giờ hallucinate (chỉ trích câu gốc)
    - Độ bao phủ cao (mỗi phần văn bản đều có đại diện)
    - Cân bằng (không ưu tiên phần đầu)
    """
    try:
        result = service.summarize_chunked(
            text=request.text,
            sentences_per_chunk=3,
            sentences_per_extraction=1
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chunked summarization failed: {str(e)}"
        )


@router.post("/smart", response_model=dict)
async def summarize_smart(
    request: SummarizationRequest,
    service: ExtractiveSummarizationService = Depends(get_extractive_service)
) -> dict:
    """
    Tóm tắt thông minh tự động (Smart Extractive) - **RECOMMENDED!**
    
    **Tự động tính số câu trích xuất dựa trên tỷ lệ 30%.**
    
    Ví dụ:
    - Văn bản 10 câu → Trích xuất 3 câu (30%)
    - Văn bản 20 câu → Trích xuất 6 câu (30%)
    
    **Ưu điểm:**
    - Không cần config thủ công
    - Độ bao phủ luôn cân đối
    - Không hallucinate (ZERO risk)
    - Có Sentence Windowing (giữ context)
    """
    try:
        result = service.summarize_by_ratio(
            text=request.text,
            ratio=0.3,  # 30% - tỷ lệ vàng
            min_sentences=1,
            max_sentences=10
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Smart summarization failed: {str(e)}"
        )


@router.get("/extractive/info", response_model=dict)
async def get_extractive_info(
    service: ExtractiveSummarizationService = Depends(get_extractive_service)
) -> dict:
    """Lấy thông tin về model trích xuất"""
    return service.get_model_info()


@router.post("/hybrid", response_model=dict)
async def summarize_hybrid(
    request: SummarizationRequest,
    service: HybridSummarizationService = Depends(get_hybrid_service)
) -> dict:
    """
    Tóm tắt lai (Hybrid Summarization) - Tốt nhất cho tiếng Việt!
    
    **Pipeline:**
    1. **Stage 1 (PhoBERT)**: Trích xuất câu quan trọng → An toàn, không bịa
    2. **Stage 2 (ViT5)**: Viết lại mượt mà → Văn phong tự nhiên
    
    **Ưu điểm:**
    - Ít hallucinate hơn ViT5 thuần (vì input ngắn)
    - Mượt mà hơn PhoBERT thuần
    - Độ bao phủ tốt
    
    Lần đầu gọi sẽ load 2 models (~400MB + ~900MB).
    """
    try:
        result = service.summarize(
            text=request.text,
            max_length=request.max_length,
            min_length=request.min_length
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hybrid summarization failed: {str(e)}"
        )


@router.get("/hybrid/info", response_model=dict)
async def get_hybrid_info(
    service: HybridSummarizationService = Depends(get_hybrid_service)
) -> dict:
    """Lấy thông tin về model hybrid"""
    return service.get_model_info()


# ==================== BARTpho Endpoints ====================

@router.post("/bartpho", response_model=dict)
async def summarize_bartpho(
    request: SummarizationRequest,
    service: BARTphoService = Depends(get_bartpho_service)
) -> dict:
    """
    Tóm tắt tiếng Việt với BARTpho (VinAI).
    
    **BARTpho** là model Seq2Seq của VinAI, cùng "gia đình" với PhoBERT.
    Văn phong tiếng Việt rất tự nhiên, ít hallucinate.
    
    Lần đầu gọi sẽ download model (~800MB).
    """
    try:
        raw_summary, processed_summary = service.summarize(
            text=request.text,
            max_length=request.max_length,
            min_length=request.min_length
        )
        
        return {
            "raw_summary": raw_summary,
            "final_summary": processed_summary,
            "model": "vinai/bartpho-syllable",
            "original_length": len(request.text),
            "final_summary_length": len(processed_summary),
            "language": "vi"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"BARTpho summarization failed: {str(e)}"
        )


@router.post("/bartpho/paraphrase", response_model=dict)
async def paraphrase_text(
    request: SummarizationRequest,
    service: BARTphoService = Depends(get_bartpho_service)
) -> dict:
    """
    Viết lại (paraphrase) văn bản tiếng Việt.
    
    **Dùng để:**
    - Làm mượt văn bản
    - Sửa lỗi ngữ pháp
    - Viết lại với văn phong tự nhiên hơn
    
    Input có thể là các câu rời rạc, output sẽ là đoạn văn liền mạch.
    """
    try:
        paraphrased = service.paraphrase(
            text=request.text,
            max_length=request.max_length,
            min_length=request.min_length
        )
        
        return {
            "original_text": request.text,
            "paraphrased_text": paraphrased,
            "model": "vinai/bartpho-syllable",
            "method": "paraphrase",
            "original_length": len(request.text),
            "paraphrased_length": len(paraphrased)
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Paraphrasing failed: {str(e)}"
        )


@router.post("/hybrid-bartpho", response_model=dict)
async def summarize_hybrid_bartpho(
    request: SummarizationRequest,
    extractive_service: ExtractiveSummarizationService = Depends(get_extractive_service),
    bartpho_service: BARTphoService = Depends(get_bartpho_service)
) -> dict:
    """
    Tóm tắt Hybrid: PhoBERT (Extractive) + BARTpho (Fusion).
    
    **Pipeline 2 bước:**
    1. **PhoBERT**: Trích xuất câu quan trọng → Không bịa thông tin
    2. **BARTpho**: Viết lại thành đoạn văn mượt mà → Văn phong tự nhiên
    
    **Ưu điểm so với PhoBERT + mT5:**
    - Cùng "hệ sinh thái" VinAI → Tương thích tốt
    - Văn phong Việt chuẩn hơn
    - Nhẹ hơn (~400MB + ~800MB = ~1.2GB tổng)
    """
    try:
        # Stage 1: PhoBERT extractive
        extractive_result = extractive_service.summarize_by_ratio(
            text=request.text,
            ratio=0.4,  # Extract 40% important sentences
            min_sentences=2,
            max_sentences=5
        )
        
        extracted_sentences = extractive_result.get("extracted_sentences", [])
        extractive_summary = extractive_result.get("summary", "")
        
        # Stage 2: BARTpho fusion
        if extracted_sentences:
            fused_summary = bartpho_service.fuse_sentences(
                sentences=extracted_sentences,
                max_length=request.max_length,
                min_length=request.min_length
            )
        else:
            fused_summary = extractive_summary
        
        return {
            "stage1_extractive": extractive_summary,
            "stage1_sentences": extracted_sentences,
            "stage1_model": "vinai/phobert-base",
            "final_summary": fused_summary,
            "final_model": "vinai/bartpho-syllable",
            "pipeline": "PhoBERT (extractive) → BARTpho (fusion)",
            "original_length": len(request.text),
            "extractive_length": len(extractive_summary),
            "final_length": len(fused_summary),
            "hallucination_risk": "LOW (content grounded in extracted sentences)"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hybrid BARTpho summarization failed: {str(e)}"
        )


@router.get("/bartpho/info", response_model=dict)
async def get_bartpho_info(
    service: BARTphoService = Depends(get_bartpho_service)
) -> dict:
    """Lấy thông tin về model BARTpho"""
    return service.get_model_info()


# ==================== Qwen Endpoints ====================

@router.post("/qwen/fuse", response_model=dict)
async def fuse_with_qwen(
    request: SummarizationRequest,
    service: QwenService = Depends(get_qwen_service)
) -> dict:
    """
    Nối các câu rời rạc thành đoạn văn mượt mà với Qwen.
    
    **Input:** Các câu cách nhau bởi dấu chấm.
    **Output:** Đoạn văn liền mạch, tự nhiên.
    
    Qwen hiểu instruction rất tốt, sẽ tự thêm từ nối phù hợp.
    Model nhẹ (~500MB), chạy được trên CPU.
    """
    try:
        # Split input into sentences
        sentences = [s.strip() for s in request.text.split('.') if s.strip()]
        
        fused_text = service.fuse_sentences(sentences)
        
        return {
            "original_sentences": sentences,
            "fused_text": fused_text,
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "method": "instruction-based fusion",
            "original_length": len(request.text),
            "fused_length": len(fused_text)
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Qwen fusion failed: {str(e)}"
        )


@router.post("/qwen/paraphrase", response_model=dict)
async def paraphrase_with_qwen(
    request: SummarizationRequest,
    service: QwenService = Depends(get_qwen_service)
) -> dict:
    """
    Viết lại văn bản mượt mà hơn với Qwen.
    
    **Qwen 2.5** là LLM nhẹ nhưng thông minh, hiểu tiếng Việt tốt.
    Sẽ giữ nguyên thông tin nhưng viết lại với văn phong tự nhiên hơn.
    """
    try:
        paraphrased = service.paraphrase(request.text)
        
        return {
            "original_text": request.text,
            "paraphrased_text": paraphrased,
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "method": "instruction-based paraphrase",
            "original_length": len(request.text),
            "paraphrased_length": len(paraphrased)
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Qwen paraphrase failed: {str(e)}"
        )


@router.post("/hybrid-qwen", response_model=dict)
async def summarize_hybrid_qwen(
    request: SummarizationRequest,
    extractive_service: ExtractiveSummarizationService = Depends(get_extractive_service),
    qwen_service: QwenService = Depends(get_qwen_service)
) -> dict:
    """
    Tóm tắt Hybrid: PhoBERT (Extractive) + Qwen (Fusion).
    
    **Pipeline 2 bước:**
    1. **PhoBERT**: Trích xuất câu quan trọng → Không bịa thông tin
    2. **Qwen**: Viết lại thành đoạn văn mượt mà → Văn phong tự nhiên
    
    **Ưu điểm:**
    - Qwen hiểu instruction tốt hơn BARTpho
    - Siêu nhẹ (~400MB + ~500MB = ~900MB tổng)
    - Ít hallucinate vì input đã được lọc bởi PhoBERT
    """
    try:
        # Stage 1: PhoBERT extractive
        extractive_result = extractive_service.summarize_by_ratio(
            text=request.text,
            ratio=0.4,  # Extract 40% important sentences
            min_sentences=2,
            max_sentences=5
        )
        
        extracted_sentences = extractive_result.get("extracted_sentences", [])
        extractive_summary = extractive_result.get("summary", "")
        
        # Stage 2: Qwen fusion
        if extracted_sentences:
            fused_summary = qwen_service.fuse_sentences(extracted_sentences)
        else:
            fused_summary = extractive_summary
        
        return {
            "stage1_extractive": extractive_summary,
            "stage1_sentences": extracted_sentences,
            "stage1_model": "vinai/phobert-base",
            "final_summary": fused_summary,
            "final_model": "Qwen/Qwen2.5-0.5B-Instruct",
            "pipeline": "PhoBERT (extractive) → Qwen (fusion)",
            "original_length": len(request.text),
            "extractive_length": len(extractive_summary),
            "final_length": len(fused_summary),
            "hallucination_risk": "LOW (Qwen follows instruction + grounded in extracted sentences)"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hybrid Qwen summarization failed: {str(e)}"
        )


@router.post("/qwen", response_model=dict)
async def summarize_with_qwen(
    request: SummarizationRequest,
    service: QwenService = Depends(get_qwen_service)
) -> dict:
    """
    Tóm tắt trực tiếp với Qwen (không qua PhoBERT).
    
    **Lưu ý:** Có thể hallucinate nếu văn bản quá dài.
    Khuyến khích dùng `/summarize/hybrid-qwen` để an toàn hơn.
    """
    try:
        summary = service.summarize(request.text)
        
        return {
            "summary": summary,
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "method": "instruction-based summarization",
            "original_length": len(request.text),
            "summary_length": len(summary),
            "warning": "Direct LLM summarization may hallucinate. Use /hybrid-qwen for safer results."
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Qwen summarization failed: {str(e)}"
        )


@router.get("/qwen/info", response_model=dict)
async def get_qwen_info(
    service: QwenService = Depends(get_qwen_service)
) -> dict:
    """Lấy thông tin về model Qwen"""
    return service.get_model_info()


# ==================== Qwen 1.5B Endpoints (More Accurate) ====================

@router.post("/hybrid-qwen-1.5b", response_model=dict)
async def summarize_hybrid_qwen_15b(
    request: SummarizationRequest,
    extractive_service: ExtractiveSummarizationService = Depends(get_extractive_service),
    qwen_service: Qwen15BService = Depends(get_qwen_15b_service)
) -> dict:
    """
    Tóm tắt Hybrid: PhoBERT (Extractive) + **Qwen 1.5B** (Fusion).
    
    **Model 1.5B thông minh hơn 0.5B:**
    - Hiểu logic tốt hơn (ai làm gì)
    - Giữ chính xác chủ ngữ
    - Ít hallucinate hơn
    
    **RAM:** ~2GB (laptop chạy tốt)
    """
    try:
        # Stage 1: PhoBERT extractive
        extractive_result = extractive_service.summarize_by_ratio(
            text=request.text,
            ratio=0.4,
            min_sentences=2,
            max_sentences=5
        )
        
        extracted_sentences = extractive_result.get("extracted_sentences", [])
        extractive_summary = extractive_result.get("summary", "")
        
        # Stage 2: Qwen 1.5B fusion
        if extracted_sentences:
            fused_summary = qwen_service.fuse_sentences(extracted_sentences)
        else:
            fused_summary = extractive_summary
        
        return {
            "stage1_extractive": extractive_summary,
            "stage1_sentences": extracted_sentences,
            "stage1_model": "vinai/phobert-base",
            "final_summary": fused_summary,
            "final_model": "Qwen/Qwen2.5-1.5B-Instruct",
            "pipeline": "PhoBERT (extractive) → Qwen-1.5B (fusion)",
            "original_length": len(request.text),
            "extractive_length": len(extractive_summary),
            "final_length": len(fused_summary),
            "model_advantage": "1.5B = 3x more parameters = better reasoning",
            "hallucination_risk": "VERY LOW (smarter model + grounded input)"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hybrid Qwen 1.5B summarization failed: {str(e)}"
        )


@router.get("/qwen-1.5b/info", response_model=dict)
async def get_qwen_15b_info(
    service: Qwen15BService = Depends(get_qwen_15b_service)
) -> dict:
    """Lấy thông tin về model Qwen 1.5B"""
    return service.get_model_info()


# ==================== Qwen Strict Endpoints (Maximum Accuracy) ====================

@router.post("/hybrid-qwen-strict", response_model=dict)
async def summarize_hybrid_qwen_strict(
    request: SummarizationRequest,
    extractive_service: ExtractiveSummarizationService = Depends(get_extractive_service),
    qwen_service: QwenStrictService = Depends(get_qwen_strict_service)
) -> dict:
    """
    Tóm tắt Hybrid: PhoBERT + **Qwen Strict Prompting**.
    
    **Sử dụng Prompt nghiêm ngặt để:**
    - KHÔNG thêm thông tin mới
    - GIỮ NGUYÊN chủ ngữ hành động
    - GIỮ NGUYÊN số liệu chính xác
    - KHÔNG thêm tính từ đánh giá
    
    **Trade-off:** Ít mượt mà hơn, nhưng chính xác hơn.
    **Dùng cho:** Văn bản pháp lý, hành chính.
    """
    try:
        # Stage 1: PhoBERT extractive
        extractive_result = extractive_service.summarize_by_ratio(
            text=request.text,
            ratio=0.4,
            min_sentences=2,
            max_sentences=5
        )
        
        extracted_sentences = extractive_result.get("extracted_sentences", [])
        extractive_summary = extractive_result.get("summary", "")
        
        # Stage 2: Qwen Strict fusion
        if extracted_sentences:
            fused_summary = qwen_service.fuse_sentences(extracted_sentences)
        else:
            fused_summary = extractive_summary
        
        return {
            "stage1_extractive": extractive_summary,
            "stage1_sentences": extracted_sentences,
            "stage1_model": "vinai/phobert-base",
            "final_summary": fused_summary,
            "final_model": "Qwen/Qwen2.5-0.5B-Instruct (STRICT MODE)",
            "pipeline": "PhoBERT (extractive) → Qwen-Strict (fusion)",
            "original_length": len(request.text),
            "extractive_length": len(extractive_summary),
            "final_length": len(fused_summary),
            "strict_rules": [
                "KHÔNG thêm thông tin mới",
                "GIỮ NGUYÊN chủ ngữ",
                "GIỮ NGUYÊN số liệu"
            ],
            "hallucination_risk": "MINIMAL (strict prompting)"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hybrid Qwen Strict summarization failed: {str(e)}"
        )


@router.get("/qwen-strict/info", response_model=dict)
async def get_qwen_strict_info(
    service: QwenStrictService = Depends(get_qwen_strict_service)
) -> dict:
    """Lấy thông tin về model Qwen Strict"""
    return service.get_model_info()


# ==================== Comparison Endpoint (All 3 Variants) ====================

@router.post("/compare-qwen", response_model=dict)
async def compare_all_qwen_variants(
    request: SummarizationRequest,
    extractive_service: ExtractiveSummarizationService = Depends(get_extractive_service),
    qwen_05b: QwenService = Depends(get_qwen_service),
    qwen_15b: Qwen15BService = Depends(get_qwen_15b_service),
    qwen_strict: QwenStrictService = Depends(get_qwen_strict_service)
) -> dict:
    """
    **SO SÁNH TẤT CẢ 3 PHIÊN BẢN QWEN** trong một request.
    
    Sử dụng cùng input từ PhoBERT, so sánh kết quả của:
    1. **Qwen 0.5B** - Nhẹ, nhanh, mượt nhưng có thể sai logic
    2. **Qwen 1.5B** - Thông minh hơn, ít sai hơn
    3. **Qwen Strict** - Prompt chặt, chính xác nhất nhưng ít mượt
    
    **⚠️ Lưu ý:** Lần đầu gọi sẽ download cả 3 models (~4GB tổng).
    """
    try:
        # Stage 1: PhoBERT extractive (shared)
        extractive_result = extractive_service.summarize_by_ratio(
            text=request.text,
            ratio=0.4,
            min_sentences=2,
            max_sentences=5
        )
        
        extracted_sentences = extractive_result.get("extracted_sentences", [])
        extractive_summary = extractive_result.get("summary", "")
        
        # Stage 2: All 3 Qwen variants
        results = {
            "input": {
                "original_text": request.text,
                "original_length": len(request.text)
            },
            "stage1_phobert": {
                "summary": extractive_summary,
                "sentences": extracted_sentences,
                "length": len(extractive_summary)
            }
        }
        
        # Qwen 0.5B
        if extracted_sentences:
            try:
                result_05b = qwen_05b.fuse_sentences(extracted_sentences)
                results["qwen_0.5b"] = {
                    "summary": result_05b,
                    "length": len(result_05b),
                    "model": "Qwen/Qwen2.5-0.5B-Instruct",
                    "characteristics": "Nhẹ, nhanh, mượt nhưng có thể sai chủ ngữ/thêm thông tin"
                }
            except Exception as e:
                results["qwen_0.5b"] = {"error": str(e)}
        
        # Qwen 1.5B
        if extracted_sentences:
            try:
                result_15b = qwen_15b.fuse_sentences(extracted_sentences)
                results["qwen_1.5b"] = {
                    "summary": result_15b,
                    "length": len(result_15b),
                    "model": "Qwen/Qwen2.5-1.5B-Instruct",
                    "characteristics": "Thông minh hơn, giữ chính xác chủ ngữ, ít hallucinate"
                }
            except Exception as e:
                results["qwen_1.5b"] = {"error": str(e)}
        
        # Qwen Strict
        if extracted_sentences:
            try:
                result_strict = qwen_strict.fuse_sentences(extracted_sentences)
                results["qwen_strict"] = {
                    "summary": result_strict,
                    "length": len(result_strict),
                    "model": "Qwen/Qwen2.5-0.5B-Instruct (STRICT)",
                    "characteristics": "Prompt nghiêm ngặt, chính xác nhất, ít mượt hơn"
                }
            except Exception as e:
                results["qwen_strict"] = {"error": str(e)}
        
        results["comparison_guide"] = {
            "fluency": "0.5B > 1.5B > Strict (mượt mà giảm dần)",
            "accuracy": "Strict > 1.5B > 0.5B (chính xác tăng dần)",
            "speed": "0.5B > Strict > 1.5B (nhanh giảm dần)",
            "recommendation": {
                "chat_app": "0.5B (nhanh, mượt)",
                "news_summary": "1.5B (cân bằng)",
                "legal_docs": "Strict (chính xác cao)"
            }
        }
        
        return results
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Comparison failed: {str(e)}"
        )
