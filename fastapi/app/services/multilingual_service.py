"""
Vietnamese Summarization Service - Using VietAI ViT5
Fine-tuned specifically for Vietnamese news summarization
"""

import logging
from typing import Optional, Tuple

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from app.utils.text_processor import TextProcessor, get_text_processor

logger = logging.getLogger(__name__)


class MultilingualSummarizationService:
    """
    Vietnamese summarization service using VietAI/vit5-base-vietnews-summarization.
    
    ViT5 is a Vietnamese T5 model fine-tuned by VietAI specifically for
    Vietnamese news article summarization.
    
    Note: Despite the class name "Multilingual", this model is optimized
    for Vietnamese. For English, use the main SummarizationService with BART.
    """
    
    MODEL_NAME = "VietAI/vit5-base-vietnews-summarization"
    
    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._text_processor: TextProcessor = get_text_processor()
        
        logger.info(f"ViT5 SummarizationService initialized. Device: {self._device}")
    
    def _load_model(self) -> None:
        """Lazy load ViT5 model"""
        if self._model is None:
            logger.info(f"Loading {self.MODEL_NAME}... (this may take 1-2 minutes)")
            self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.MODEL_NAME)
            self._model.to(self._device)
            self._model.eval()
            logger.info(f"{self.MODEL_NAME} loaded successfully!")
    
    def summarize(
        self,
        text: str,
        max_length: int = 150,
        min_length: int = 30,
        language: str = "vi"
    ) -> Tuple[str, str]:
        """
        Summarize Vietnamese text.
        
        Args:
            text: Input Vietnamese text to summarize
            max_length: Maximum summary length (in tokens)
            min_length: Minimum summary length (in tokens)
            language: Language hint (vi for Vietnamese)
            
        Returns:
            Tuple[str, str]: (raw_summary, processed_summary)
        """
        self._load_model()
        
        # PRE-PROCESSING
        cleaned_text = self._text_processor.preprocess(text)
        
        # ViT5 fine-tuned model - just use the text directly (no prefix needed)
        input_text = cleaned_text
        
        inputs = self._tokenizer(
            input_text,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        ).to(self._device)
        
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_length=max_length,
                min_length=min_length,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
        
        raw_summary = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # POST-PROCESSING (lighter for Vietnamese)
        processed_summary = raw_summary.strip()
        
        # Ensure proper ending
        if processed_summary and processed_summary[-1] not in '.!?':
            processed_summary += '.'
        
        return raw_summary, processed_summary
    
    def get_model_info(self) -> dict:
        """Return information about the model"""
        return {
            "model_name": self.MODEL_NAME,
            "description": "ViT5 - Vietnamese T5 fine-tuned for news summarization by VietAI",
            "supported_languages": ["vi"],
            "model_size": "~900MB",
            "loaded": self._model is not None,
            "organization": "VietAI (Vietnamese AI Community)"
        }


# Singleton instance for dependency injection
_multilingual_service: Optional[MultilingualSummarizationService] = None


def get_multilingual_service() -> MultilingualSummarizationService:
    """Get or create MultilingualSummarizationService singleton"""
    global _multilingual_service
    if _multilingual_service is None:
        _multilingual_service = MultilingualSummarizationService()
    return _multilingual_service
