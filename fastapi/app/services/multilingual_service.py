"""
Multilingual Summarization Service - Using Google mT5-base
Supports 101 languages including Vietnamese and English
"""

import logging
from typing import Optional, Tuple

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from app.utils.text_processor import TextProcessor, get_text_processor

logger = logging.getLogger(__name__)


class MultilingualSummarizationService:
    """
    Multilingual summarization service using google/mt5-base.
    
    mT5 (Multilingual T5) is trained on 101 languages including:
    - Vietnamese (vi)
    - English (en)
    - And 99 other languages
    
    This service complements the main BART-based service for
    non-English text summarization.
    """
    
    MODEL_NAME = "google/mt5-base"
    
    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._text_processor: TextProcessor = get_text_processor()
        
        logger.info(f"MultilingualSummarizationService initialized. Device: {self._device}")
    
    def _load_model(self) -> None:
        """Lazy load mT5-base model"""
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
        language: str = "auto"
    ) -> Tuple[str, str]:
        """
        Summarize text in any of 101 supported languages.
        
        Args:
            text: Input text to summarize
            max_length: Maximum summary length (in tokens)
            min_length: Minimum summary length (in tokens)
            language: Language hint (auto, vi, en, etc.) - currently unused
            
        Returns:
            Tuple[str, str]: (raw_summary, processed_summary)
        """
        self._load_model()
        
        # PRE-PROCESSING
        cleaned_text = self._text_processor.preprocess(text)
        
        # mT5 uses task prefix for summarization
        # Using "summarize:" prefix which works for multilingual
        input_text = f"summarize: {cleaned_text}"
        
        inputs = self._tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
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
        
        # POST-PROCESSING
        entities = self._text_processor.extract_entities(text)
        processed_summary = self._text_processor.postprocess(raw_summary, entities)
        
        return raw_summary, processed_summary
    
    def get_model_info(self) -> dict:
        """Return information about the model"""
        return {
            "model_name": self.MODEL_NAME,
            "description": "Multilingual T5 (mT5) - supports 101 languages",
            "supported_languages": ["vi", "en", "zh", "ja", "ko", "th", "id", "and 94 more..."],
            "model_size": "~900MB",
            "loaded": self._model is not None
        }


# Singleton instance for dependency injection
_multilingual_service: Optional[MultilingualSummarizationService] = None


def get_multilingual_service() -> MultilingualSummarizationService:
    """Get or create MultilingualSummarizationService singleton"""
    global _multilingual_service
    if _multilingual_service is None:
        _multilingual_service = MultilingualSummarizationService()
    return _multilingual_service
