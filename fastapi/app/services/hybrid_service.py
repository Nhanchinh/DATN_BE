"""
Hybrid Summarization Service - Extractive + Abstractive
Best of both worlds: Safety of extraction + Fluency of abstraction
Now using mT5-XLSum instead of ViT5!
"""

import logging
from typing import Optional, Dict

from app.services.extractive_service import ExtractiveSummarizationService, get_extractive_service
from app.services.xlsum_service import XLSumService, get_xlsum_service

logger = logging.getLogger(__name__)


class HybridSummarizationService:
    """
    Hybrid summarization combining extractive and abstractive approaches.
    
    Pipeline:
    1. EXTRACT: Use PhoBERT to extract important sentences (safe, no hallucination)
    2. REWRITE: Use mT5-XLSum to polish and make text more fluent
    
    Benefits:
    - Reduced hallucination risk (mT5-XLSum is more diverse than ViT5)
    - Better coverage than pure extraction
    - More fluent output than pure extraction
    """
    
    def __init__(self):
        self._extractive_service = None
        self._abstractive_service = None
        
        logger.info("HybridSummarizationService initialized (with mT5-XLSum)")
    
    def _load_services(self) -> None:
        """Lazy load both services"""
        if self._extractive_service is None:
            self._extractive_service = get_extractive_service()
        if self._abstractive_service is None:
            self._abstractive_service = get_xlsum_service()  # Changed from ViT5 to XLSum
    
    def summarize(
        self,
        text: str,
        extract_ratio: float = 0.4,
        max_length: int = 150,
        min_length: int = 30
    ) -> Dict:
        """
        Hybrid summarization: Extract then Rewrite.
        
        Args:
            text: Input Vietnamese text
            extract_ratio: Ratio of sentences to extract in Stage 1
            max_length: Max length for Stage 2 rewrite
            min_length: Min length for Stage 2 rewrite
            
        Returns:
            Dict with all intermediate and final results
        """
        self._load_services()
        
        logger.info("Starting hybrid summarization (PhoBERT + mT5-XLSum)...")
        
        # =====================
        # STAGE 1: EXTRACT
        # =====================
        logger.info("Stage 1: Extracting important sentences with PhoBERT...")
        
        extracted_summary, extracted_sentences = self._extractive_service.summarize(
            text=text,
            ratio=extract_ratio
        )
        
        logger.info(f"Extracted {len(extracted_sentences)} sentences")
        
        # If extraction is very short, return as-is
        if len(extracted_summary) < 50:
            return {
                "stage1_extracted": extracted_summary,
                "stage1_sentences": extracted_sentences,
                "stage2_rewritten": extracted_summary,
                "final_summary": extracted_summary,
                "method": "hybrid (extraction only - text too short)",
                "original_length": len(text),
                "final_length": len(extracted_summary)
            }
        
        # =====================
        # STAGE 2: REWRITE with mT5-XLSum
        # =====================
        logger.info("Stage 2: Rewriting with mT5-XLSum for fluency...")
        
        # Use mT5-XLSum to polish the extracted sentences
        raw_rewrite, polished_summary = self._abstractive_service.summarize(
            text=extracted_summary,
            max_length=max_length,
            min_length=min_length
        )
        
        logger.info("Hybrid summarization complete")
        
        return {
            "stage1_extracted": extracted_summary,
            "stage1_sentences": extracted_sentences,
            "stage1_length": len(extracted_summary),
            "stage2_rewritten": polished_summary,
            "final_summary": polished_summary,
            "method": "hybrid (PhoBERT extract + mT5-XLSum rewrite)",
            "stage2_model": "csebuetnlp/mT5_multilingual_XLSum",
            "original_length": len(text),
            "final_length": len(polished_summary),
            "compression_ratio": round(len(polished_summary) / len(text), 3)
        }
    
    def get_model_info(self) -> dict:
        """Return information about the hybrid approach"""
        return {
            "method": "Hybrid Summarization",
            "stage1_model": "vinai/phobert-base (Extractive)",
            "stage2_model": "csebuetnlp/mT5_multilingual_XLSum (Abstractive)",  # Updated
            "description": "Extract important sentences, then rewrite for fluency",
            "hallucination_risk": "LOW (mT5-XLSum is more robust than ViT5)",
            "supported_languages": ["vi", "en", "and 43 more"]
        }


# Singleton instance
_hybrid_service: Optional[HybridSummarizationService] = None


def get_hybrid_service() -> HybridSummarizationService:
    """Get or create HybridSummarizationService singleton"""
    global _hybrid_service
    if _hybrid_service is None:
        _hybrid_service = HybridSummarizationService()
    return _hybrid_service
