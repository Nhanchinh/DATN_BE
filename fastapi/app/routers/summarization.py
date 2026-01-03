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


router = APIRouter(prefix="/summarize", tags=["summarization"])


@router.post("", response_model=SummarizationResponse, summary="[BART-cnn] Tóm tắt tiếng Anh")
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


@router.post("/balanced", response_model=dict, summary="[BART-cnn] Tóm tắt cân bằng theo topic")
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


@router.post("/detailed", response_model=dict, summary="[BART-cnn] Phân tích chi tiết")
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


@router.post("/raw", response_model=dict, summary="[BART-cnn] Tóm tắt thô (không hậu xử lý)")
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


@router.post("/refine", response_model=RefineResponse, summary="[POST-PROCESS] Tinh chỉnh văn bản")
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


@router.post("/multilingual", response_model=dict, summary="[ViT5-Finetuned] Tóm tắt tiếng Việt")
async def summarize_multilingual(
    request: SummarizationRequest,
    service: MultilingualSummarizationService = Depends(get_multilingual_service)
) -> dict:
    """
    Tóm tắt tiếng Việt với ViT5 đã fine-tune.
    
    Model ViT5-base được fine-tune thêm cho tóm tắt tiếng Việt.
    
    Sử dụng model local: AI_Models/my_vit5_model (~900MB).
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
            "model": "AI_Models/my_vit5_model (finetuned ViT5)",
            "original_length": len(request.text),
            "final_summary_length": len(processed_summary),
            "language": "vi"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vietnamese summarization failed: {str(e)}"
        )


@router.get("/multilingual/info", response_model=dict, summary="[ViT5] Thông tin model")
async def get_multilingual_info(
    service: MultilingualSummarizationService = Depends(get_multilingual_service)
) -> dict:
    """Lấy thông tin về model đa ngôn ngữ"""
    return service.get_model_info()


@router.post("/extractive", response_model=dict, summary="[PhoBERT] Trích xuất câu quan trọng")
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


@router.post("/chunked", response_model=dict, summary="[PhoBERT] Trích xuất theo chunks")
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


@router.post("/smart", response_model=dict, summary="[PhoBERT] ⭐ Trích xuất thông minh (RECOMMENDED)")
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


@router.get("/extractive/info", response_model=dict, summary="[PhoBERT] Thông tin model")
async def get_extractive_info(
    service: ExtractiveSummarizationService = Depends(get_extractive_service)
) -> dict:
    """Lấy thông tin về model trích xuất"""
    return service.get_model_info()


@router.post("/hybrid", response_model=dict, summary="[PhoBERT → mT5-XLSum] Hybrid tóm tắt")
async def summarize_hybrid(
    request: SummarizationRequest,
    service: HybridSummarizationService = Depends(get_hybrid_service)
) -> dict:
    """
    Tóm tắt lai (Hybrid Summarization) - Tốt nhất cho tiếng Việt!
    
    **Pipeline:**
    1. **Stage 1 (PhoBERT)**: Trích xuất câu quan trọng → An toàn, không bịa
    2. **Stage 2 (mT5-XLSum)**: Viết lại mượt mà → Văn phong tự nhiên
    
    **Ưu điểm:**
    - Ít hallucinate hơn (mT5-XLSum đa dạng hơn ViT5)
    - Mượt mà hơn PhoBERT thuần
    - Hỗ trợ 44+ ngôn ngữ
    
    Lần đầu gọi sẽ load 2 models (~400MB + ~1.2GB).
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


@router.get("/hybrid/info", response_model=dict, summary="[Hybrid] Thông tin pipeline")
async def get_hybrid_info(
    service: HybridSummarizationService = Depends(get_hybrid_service)
) -> dict:
    """Lấy thông tin về model hybrid"""
    return service.get_model_info()


# ==================== BARTpho Endpoints ====================

@router.post("/bartpho", response_model=dict, summary="[BARTpho] Tóm tắt tiếng Việt")
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


@router.post("/bartpho/paraphrase", response_model=dict, summary="[BARTpho] Viết lại (paraphrase)")
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


@router.post("/hybrid-bartpho", response_model=dict, summary="[PhoBERT → BARTpho] Hybrid Việt")
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


@router.post("/hybrid-vit5", response_model=dict, summary="[PhoBERT → ViT5] ⭐ Hybrid BEST (RECOMMENDED)")
async def summarize_hybrid_vit5(
    request: SummarizationRequest,
    extractive_service: ExtractiveSummarizationService = Depends(get_extractive_service),
    vit5_service: MultilingualSummarizationService = Depends(get_multilingual_service)
) -> dict:
    """
    Tóm tắt Hybrid: PhoBERT (Extractive) + ViT5 Finetuned (Rewrite).
    
    **Pipeline 2 bước (Chia để trị):**
    1. **PhoBERT**: Trích xuất câu quan trọng → Không bịa thông tin
    2. **ViT5 Local**: Làm mượt TỪNG CÂU riêng lẻ → Giữ nguyên tất cả ý
    
    **Ưu điểm:**
    - Xử lý từng câu đúng với cách model được train
    - KHÔNG mất ý (mỗi câu được xử lý riêng)
    - Văn phong tiếng Việt chuẩn
    
    Lần đầu gọi sẽ load 2 models (~400MB + ~900MB = ~1.3GB).
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
        
        # Stage 2: ViT5 finetuned - XỬ LÝ TỪNG CÂU RIÊNG LẺ (Chia để trị)
        # Đây là cách đúng vì model được train với input là 1 câu
        if extracted_sentences:
            final_summary = vit5_service.smooth_sentences(extracted_sentences)
        else:
            final_summary = extractive_summary
        
        return {
            "stage1_extracted": extractive_summary,
            "stage1_sentences": extracted_sentences,
            "stage1_length": len(extractive_summary),
            "stage1_model": "vinai/phobert-base",
            "stage2_rewritten": final_summary,
            "final_summary": final_summary,
            "stage2_model": "AI_Models/my_vit5_model (finetuned ViT5)",
            "pipeline": "PhoBERT (extractive) → ViT5 (smooth từng câu)",
            "method": "Chia để trị - xử lý từng câu riêng lẻ",
            "original_length": len(request.text),
            "final_length": len(final_summary),
            "compression_ratio": round(len(final_summary) / len(request.text), 3) if len(request.text) > 0 else 0,
            "hallucination_risk": "VERY LOW (PhoBERT grounds + ViT5 smooths each sentence)"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hybrid ViT5 summarization failed: {str(e)}"
        )


@router.get("/bartpho/info", response_model=dict, summary="[BARTpho] Thông tin model")
async def get_bartpho_info(
    service: BARTphoService = Depends(get_bartpho_service)
) -> dict:
    return service.get_model_info()

