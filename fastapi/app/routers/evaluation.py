"""
Evaluation Router - API endpoints cho evaluation metrics
Đánh giá single/batch, compare models, history tracking
"""

import logging
from typing import List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from app.repositories.evaluation_repository import get_evaluation_repository
from app.schemas.evaluation import (
    BatchProgressResponse,
    CompareModelsRequest,
    CompareModelsResponse,
    DatasetsResponse,
    EvaluateBatchRequest,
    EvaluateBatchResponse,
    EvaluateSingleRequest,
    EvaluateSingleResponse,
    EvaluationHistoryResponse,
    EvaluationHistoryItem,
    EvaluationMetricsResponse,
    ModelComparisonItem,
    OverallMetricsResponse,
    DatasetInfo,
)
from app.services.dataset_service import get_dataset_service
from app.services.evaluation_service import get_evaluation_service
from app.utils.dependencies import get_current_user

# Import summarization services
from app.services.extractive_service import get_extractive_service
from app.services.multilingual_service import get_multilingual_service
from app.services.bartpho_service import get_bartpho_service
from app.services.hybrid_service import get_hybrid_service
from app.services.vit5_paraphrase_service import get_vit5_paraphrase_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/evaluate", tags=["evaluation"])


# Helper function để get summarization service theo model name
def get_summarization_service(model_name: str):
    """
    Route model name to appropriate service.
    
    Supported models:
    - extractive, extractive_smart, extractive_chunked: PhoBERT
    - vit5, multilingual: ViT5 Local Finetuned
    - bartpho: BARTpho (VinAI)
    - hybrid: PhoBERT + mT5-XLSum
    - hybrid_bartpho: PhoBERT + BARTpho (VinAI ecosystem)
    - hybrid_vit5: PhoBERT + ViT5 Local ⭐ BEST!
    - hybrid_paraphrase: PhoBERT + ViT5 Paraphrase 🔥 NEW! (Smooth)
    """
    model_map = {
        "extractive": get_extractive_service(),
        "extractive_smart": get_extractive_service(),
        "extractive_chunked": get_extractive_service(),
        "vit5": get_multilingual_service(),
        "multilingual": get_multilingual_service(),
        "bartpho": get_bartpho_service(),
        "hybrid": get_hybrid_service(),
        "hybrid_bartpho": (get_extractive_service(), get_bartpho_service()),
        "hybrid_vit5": (get_extractive_service(), get_multilingual_service()),
        "hybrid_paraphrase": (get_extractive_service(), get_vit5_paraphrase_service()),  # NEW!
    }
    
    service = model_map.get(model_name)
    if not service:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown model: {model_name}. Supported: {list(model_map.keys())}"
        )
    
    return service, model_name


async def generate_summary(model_name: str, text: str, max_length: int = 150) -> str:
    """
    Generate summary using specified model.
    
    Args:
        model_name: Name of the model
        text: Input text
        max_length: Max length for summary
        
    Returns:
        Generated summary text
    """
    service_result = get_summarization_service(model_name)
    
    # Handle hybrid models (tuple of 2 services) vs single service
    if isinstance(service_result[0], tuple):
        # Hybrid models return tuple of (extractive_service, rewrite_service)
        services = service_result[0]
        is_hybrid_double = True
    else:
        # Single service
        service = service_result[0]
        is_hybrid_double = False
    
    # Call appropriate summarization method based on model type
    if model_name in ["extractive_smart", "extractive"]:
        # Use summarize_by_ratio (the ACTUAL method name)
        result = service.summarize_by_ratio(
            text=text,
            ratio=0.3,  # 30% ratio 
            min_sentences=1,
            max_sentences=10
        )
        return result["summary"]
    
    elif model_name == "extractive_chunked":
        result = service.summarize_chunked(
            text=text,
            sentences_per_chunk=3,
            sentences_per_extraction=1
        )
        return result["summary"]
    
    elif model_name in ["vit5", "multilingual"]:
        raw_summary, processed_summary = service.summarize(
            text=text,
            max_length=max_length,
            min_length=30
        )
        return processed_summary
    
    elif model_name == "bartpho":
        raw_summary, processed_summary = service.summarize(
            text=text,
            max_length=max_length,
            min_length=30
        )
        return processed_summary
    
    elif model_name == "hybrid":
        result = service.summarize(
            text=text,
            max_length=max_length,
            min_length=30
        )
        # Hybrid returns dict with "final_summary" key
        return result.get("final_summary", result.get("summary", ""))
    
    elif model_name == "hybrid_bartpho":
        # PhoBERT (extractive) + BARTpho (rewrite CẢ ĐOẠN với context)
        extractive_service, bartpho_service = services
        
        # Stage 1: PhoBERT extractive
        # IMPORTANT: Tăng ratio để bao phủ đủ ý (coverage)
        extractive_result = extractive_service.summarize_by_ratio(
            text=text,
            ratio=0.6,  # 60% thay vì 40% - BẮT được nhiều ý hơn
            min_sentences=3,  # Tối thiểu 3 câu
            max_sentences=8   # Tối đa 8 câu
        )
        extracted_sentences = extractive_result.get("extracted_sentences", [])
        
        # Stage 2: BARTpho - GHÉP CẢ ĐOẠN (FIX Context Blindness!)
        # CRITICAL: Không dùng fuse_sentences nữa, cho BARTpho xử lý cả đoạn văn
        if extracted_sentences:
            # Ghép tất cả câu extract thành 1 đoạn văn
            combined_text = " ".join(extracted_sentences)
            
            # BARTpho viết lại CẢ ĐOẠN với context đầy đủ
            raw_summary, processed_summary = bartpho_service.summarize(
                text=combined_text,
                max_length=max_length,
                min_length=30
            )
            return processed_summary
        else:
            return extractive_result.get("summary", "")
    
    elif model_name == "hybrid_vit5":
        # PhoBERT (extractive) + ViT5 Local (smooth CẢ ĐOẠN với context) - BEST!
        extractive_service, vit5_service = services
        
        # Stage 1: PhoBERT extractive
        # IMPORTANT: Tăng ratio để bao phủ đủ ý (coverage)
        extractive_result = extractive_service.summarize_by_ratio(
            text=text,
            ratio=0.6,  # 60% thay vì 40% - BẮT được nhiều ý hơn
            min_sentences=3,  # Tối thiểu 3 câu
            max_sentences=8   # Tối đa 8 câu (cho văn bản dài)
        )
        extracted_sentences = extractive_result.get("extracted_sentences", [])
        
        # Stage 2: ViT5 - GHÉP TẤT CẢ CÂU LẠI (FIX Context Blindness!)
        # CRITICAL: Không xử lý từng câu nữa, ghép thành đoạn văn để giữ ngữ cảnh
        if extracted_sentences:
            # Ghép tất cả câu extract thành 1 đoạn văn
            combined_text = " ".join(extracted_sentences)
            
            # ViT5 xử lý CẢ ĐOẠN với đầy đủ ngữ cảnh
            # Không bị "mù ngữ cảnh" nữa!
            raw_summary, final_summary = vit5_service.summarize(
                text=combined_text,
                max_length=max_length,
                min_length=30
            )
            return final_summary
        else:
            return extractive_result.get("summary", "")
    
    elif model_name == "hybrid_paraphrase":
        # PhoBERT (segmentation extraction) + ViT5 Paraphrase (smooth with chunking) - NEW!
        extractive_service, vit5_paraphrase_service = services
        
        # Stage 1: PhoBERT segmentation (2-5-2 distribution)
        extractive_result = extractive_service.extract_with_segmentation(
            text=text,
            quotas={"intro": 2, "body": 5, "conclusion": 2},
            total_sentences=9
        )
        extracted_sentences = extractive_result.get("extracted_sentences", [])
        
        # Stage 2: ViT5 Paraphrase with chunking (3 sentences per chunk)
        if extracted_sentences:
            final_summary = vit5_paraphrase_service.paraphrase_sentences(
                sentences=extracted_sentences,
                chunk_size=3,
                max_length=max_length,
                min_length=20
            )
            return final_summary
        else:
            return extractive_result.get("summary", "")
    
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unsupported model type: {model_name}"
        )


# ==================== Endpoints ====================

@router.post("/single", response_model=EvaluateSingleResponse)
async def evaluate_single(
    request: EvaluateSingleRequest,
    current_user: dict = Depends(get_current_user),
    eval_service = Depends(get_evaluation_service),
    eval_repo = Depends(get_evaluation_repository)
):
    """
    Đánh giá 1 văn bản với model chỉ định.
    
    **Flow**:
    1. Generate summary từ model
    2. Calculate metrics (ROUGE, BLEU, BERTScore)
    3. Save vào MongoDB
    4. Return metrics
    
    **Auth**: Required (user token)
    """
    try:
        logger.info(f"Single evaluation started by user {current_user['_id']} with model {request.model_name}")
        
        # Step 1: Generate summary
        generated_summary = await generate_summary(
            request.model_name,
            request.text,
            request.max_length
        )
        
        # Step 2: Calculate metrics
        metrics = await eval_service.evaluate_single(
            prediction=generated_summary,
            reference=request.reference_summary,
            calculate_bert=not request.skip_bert  # Skip nếu user yêu cầu
        )
        
        # Step 3: Save to MongoDB
        evaluation_id = await eval_repo.save_evaluation_result(
            user_id=str(current_user['_id']),
            model_name=request.model_name,
            dataset_name="manual",  # Single evaluations are marked as manual
            summary_data=[{
                "input_text": request.text,
                "reference_summary": request.reference_summary,
                "generated_summary": generated_summary,
                "metrics": metrics
            }],
            overall_metrics={
                "avg_rouge1": metrics['rouge1'],
                "avg_rouge2": metrics['rouge2'],
                "avg_rougeL": metrics['rougeL'],
                "avg_bleu": metrics['bleu'],
                "avg_bert_score": metrics['bert_score'],
                "avg_processing_time_ms": metrics['processing_time_ms'],
                "total_samples": 1
            },
            status="completed"
        )
        
        logger.info(f"Single evaluation completed: {evaluation_id}")
        
        return EvaluateSingleResponse(
            generated_summary=generated_summary,
            metrics=EvaluationMetricsResponse(**metrics),
            evaluation_id=evaluation_id
        )
        
    except Exception as e:
        logger.error(f"Single evaluation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {str(e)}"
        )


async def run_batch_evaluation(
    evaluation_id: str,
    user_id: str,
    model_name: str,
    dataset_name: str,
    sample_size: int,
    max_length: int,
    eval_service,
    eval_repo,
    dataset_service
):
    """
    Background task for batch evaluation.
    
    This runs asynchronously to avoid HTTP timeout.
    """
    try:
        logger.info(f"Batch evaluation {evaluation_id} started")
        
        # Update status to running
        await eval_repo.update_evaluation_status(evaluation_id, "running")
        
        # Load dataset
        if dataset_name == "vietnews":
            articles, references = dataset_service.get_test_split(
                sample_size=sample_size,
                shuffle=True
            )
        else:
            # Custom dataset (future implementation)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Custom datasets not  yet implemented"
            )
        
        # Generate summaries for all articles
        predictions = []
        summary_data = []
        
        for i, (article, reference) in enumerate(zip(articles, references)):
            logger.info(f"Processing sample {i+1}/{len(articles)}")
            
            try:
                # Generate summary
                summary = await generate_summary(model_name, article, max_length)
                predictions.append(summary)
                
                # Calculate metrics for this sample
                metrics = await eval_service.evaluate_single(
                    prediction=summary,
                    reference=reference,
                    calculate_bert=False  # Skip BERTScore for individual samples (too slow)
                )
                
                summary_data.append({
                    "input_text": article[:200] + "...",  # Truncate for storage
                    "reference_summary": reference,
                    "generated_summary": summary,
                    "metrics": metrics
                })
                
            except Exception as e:
                logger.error(f"Failed to process sample {i}: {e}")
                continue
        
        # Calculate overall metrics with BERTScore
        logger.info("Calculating overall metrics...")
        overall_metrics = await eval_service.evaluate_batch(
            predictions=predictions,
            references=references[:len(predictions)],  # Match length
            calculate_bert=True,  # Now calculate BERTScore on batch
            batch_size=16
        )
        
        # Save results
        await eval_repo.save_evaluation_result(
            user_id=user_id,
            model_name=model_name,
            dataset_name=dataset_name,
            summary_data=summary_data,
            overall_metrics=overall_metrics,
            status="completed"
        )
        
        logger.info(f"Batch evaluation {evaluation_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Batch evaluation {evaluation_id} failed: {e}")
        await eval_repo.update_evaluation_status(
            evaluation_id,
            "failed",
            error_message=str(e)
        )


@router.post("/batch", response_model=EvaluateBatchResponse)
async def evaluate_batch(
    request: EvaluateBatchRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    eval_service = Depends(get_evaluation_service),
    eval_repo = Depends(get_evaluation_repository),
    dataset_service = Depends(get_dataset_service)
):
    """
    Đánh giá batch với dataset.
    
    **Background Task**: Chạy async để tránh timeout.
    Use GET /evaluate/progress/{evaluation_id} để track progress.
    
    **Performance Note**:
    - 100 samples: ~5-10 minutes (depending on model)
    - BERTScore rất chậm, chỉ tính cho overall metrics
    - Max 3 models cùng lúc để tránh RAM overflow
    
    **Auth**: Required
    """
    try:
        logger.info(f"Batch evaluation requested by user {current_user['_id']}")
        
        # Create initial evaluation record
        evaluation_id = await eval_repo.save_evaluation_result(
            user_id=str(current_user['_id']),
            model_name=request.model_names[0],  # First model
            dataset_name=request.dataset_name,
            summary_data=[],
            overall_metrics={
                "avg_rouge1": 0.0,
                "avg_rouge2": 0.0,
                "avg_rougeL": 0.0,
                "avg_bleu": 0.0,
                "avg_bert_score": 0.0,
                "avg_processing_time_ms": 0,
                "total_samples": 0
            },
            status="queued"
        )
        
        # Queue background tasks for each model
        # CRITICAL: Run sequentially to avoid RAM overflow
        for model_name in request.model_names:
            background_tasks.add_task(
                run_batch_evaluation,
                evaluation_id,
                str(current_user['_id']),
                model_name,
                request.dataset_name,
                request.sample_size,
                request.max_length,
                eval_service,
                eval_repo,
                dataset_service
            )
        
        return EvaluateBatchResponse(
            evaluation_id=evaluation_id,
            status="queued",
            message=f"Batch evaluation queued for {len(request.model_names)} model(s). Use /evaluate/progress/{evaluation_id} to track progress."
        )
        
    except Exception as e:
        logger.error(f"Failed to queue batch evaluation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start batch evaluation: {str(e)}"
        )


@router.get("/progress/{evaluation_id}", response_model=BatchProgressResponse)
async def get_batch_progress(
    evaluation_id: str,
    current_user: dict = Depends(get_current_user),
    eval_repo = Depends(get_evaluation_repository)
):
    """
    Lấy progress của batch evaluation.
    
    Frontend nên poll endpoint này mỗi 5s để update progress bar.
    
    **Auth**: Required
    """
    evaluation = await eval_repo.get_evaluation_by_id(evaluation_id)
    
    if not evaluation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation not found"
        )
    
    # Check ownership
    if evaluation['user_id'] != str(current_user['_id']):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view this evaluation"
        )
    
    # Calculate progress
    status_value = evaluation['status']
    progress = 0
    
    if status_value == "queued":
        progress = 0
    elif status_value == "running":
        # Estimate progress based on total_samples
        total = evaluation['overall_metrics'].get('total_samples', 0)
        progress = min(90, int(total / 100 * 90)) if total > 0 else 10
    elif status_value == "completed":
        progress = 100
    elif status_value == "failed":
        progress = 0
    
    # Build response
    overall_metrics = None
    if status_value == "completed":
        overall_metrics = OverallMetricsResponse(**evaluation['overall_metrics'])
    
    return BatchProgressResponse(
        evaluation_id=evaluation_id,
        status=status_value,
        progress=progress,
        current_sample=evaluation['overall_metrics'].get('total_samples', 0),
        total_samples=evaluation['overall_metrics'].get('total_samples', 0),
        overall_metrics=overall_metrics,
        error_message=evaluation.get('error_message')
    )


@router.post("/compare", response_model=CompareModelsResponse)
async def compare_models(
    request: CompareModelsRequest,
    current_user: dict = Depends(get_current_user),
    eval_repo = Depends(get_evaluation_repository)
):
    """
    So sánh performance của nhiều models.
    
    Sử dụng MongoDB aggregation để tính average metrics từ evaluation history.
    
    **Auth**: Required
    """
    try:
        comparisons_dict = await eval_repo.compare_models(
            user_id=str(current_user['_id']),
            model_names=request.model_names,
            dataset_name=request.dataset_name,
            limit=request.limit
        )
        
        # Convert to response format
        comparisons_list = [
            ModelComparisonItem(
                model_name=model_name,
                avg_rouge1=data.get('avg_rouge1', 0.0),
                avg_rouge2=data.get('avg_rouge2', 0.0),
                avg_rougeL=data.get('avg_rougeL', 0.0),
                avg_bleu=data.get('avg_bleu', 0.0),
                avg_bert_score=data.get('avg_bert_score', 0.0),
                evaluation_count=data.get('count', 0)
            )
            for model_name, data in comparisons_dict.items()
        ]
        
        return CompareModelsResponse(
            comparisons=comparisons_list,
            dataset_name=request.dataset_name
        )
        
    except Exception as e:
        logger.error(f"Model comparison failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Comparison failed: {str(e)}"
        )


@router.get("/history", response_model=EvaluationHistoryResponse)
async def get_evaluation_history(
    limit: int = 10,
    model_name: str = None,
    dataset_name: str = None,
    current_user: dict = Depends(get_current_user),
    eval_repo = Depends(get_evaluation_repository)
):
    """
    Lấy lịch sử evaluations của user.
    
    **Query params**:
    - limit: Số lượng records (default 10)
    - model_name: Filter theo model (optional)
    - dataset_name: Filter theo dataset (optional)
    
    **Auth**: Required
    """
    try:
        evaluations = await eval_repo.get_evaluation_history(
            user_id=str(current_user['_id']),
            limit=limit,
            model_name=model_name,
            dataset_name=dataset_name
        )
        
        # Convert to response format
        history_items = [
            EvaluationHistoryItem(
                evaluation_id=eval['_id'],
                model_name=eval['model_name'],
                dataset_name=eval['dataset_name'],
                created_at=eval['created_at'],
                overall_metrics=OverallMetricsResponse(**eval['overall_metrics']),
                status=eval['status']
            )
            for eval in evaluations
        ]
        
        return EvaluationHistoryResponse(
            evaluations=history_items,
            total_count=len(history_items)
        )
        
    except Exception as e:
        logger.error(f"Failed to get evaluation history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve history: {str(e)}"
        )


@router.get("/datasets", response_model=DatasetsResponse)
async def get_available_datasets(
    current_user: dict = Depends(get_current_user),
    dataset_service = Depends(get_dataset_service)
):
    """
    Liệt kê các datasets có sẵn.
    
    **Auth**: Required
    """
    try:
        # Get VietNews info
        vietnews_info = dataset_service.get_dataset_info("vietnews")
        
        datasets = [
            DatasetInfo(**vietnews_info)
        ]
        
        # TODO: Add custom datasets from DB
        
        return DatasetsResponse(datasets=datasets)
        
    except Exception as e:
        logger.error(f"Failed to get datasets: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve datasets: {str(e)}"
        )
