"""
XLSum Summarization Service - Using mT5 Multilingual XLSum
More diverse training data = less hallucination
"""

import logging
from typing import Optional, Tuple

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logger = logging.getLogger(__name__)


class XLSumService:
    """
    Abstractive summarization using mT5 trained on XL-Sum dataset.
    
    XL-Sum is a multilingual summarization dataset covering 45 languages
    with diverse news sources (BBC, etc.). This model should be more
    robust than ViT5 which was only trained on Vietnamese news.
    """
    
    MODEL_NAME = "csebuetnlp/mT5_multilingual_XLSum"
    
    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"XLSumService initialized. Device: {self._device}")
    
    def _load_model(self) -> None:
        """Lazy load mT5-XLSum model"""
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
        min_length: int = 30
    ) -> Tuple[str, str]:
        """
        Summarize text using mT5-XLSum.
        
        Args:
            text: Input text (Vietnamese or other 45 supported languages)
            max_length: Maximum summary length in tokens
            min_length: Minimum summary length in tokens
            
        Returns:
            Tuple[str, str]: (raw_summary, processed_summary)
        """
        self._load_model()
        
        # XLSum models expect plain text input (no prefix needed)
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(self._device)
        
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_length=max_length,
                min_length=min_length,
                num_beams=5,                # Tăng từ 4 lên 5 - chính xác hơn
                length_penalty=2.0,
                early_stopping=True,
                no_repeat_ngram_size=3,
                do_sample=False,            # Tắt sampling - chỉ dùng beam search
                temperature=1.0,            # Khi do_sample=False, temperature không ảnh hưởng
                repetition_penalty=2.5,     # Giảm lặp từ - tránh output vô nghĩa
            )
        
        raw_summary = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Simple post-processing
        processed_summary = raw_summary.strip()
        if processed_summary and processed_summary[-1] not in '.!?':
            processed_summary += '.'
        
        return raw_summary, processed_summary
    
    def get_model_info(self) -> dict:
        """Return information about the model"""
        return {
            "model_name": self.MODEL_NAME,
            "description": "mT5 fine-tuned on XL-Sum dataset - 45 languages, diverse sources",
            "training_data": "XL-Sum (BBC news from 45 languages)",
            "supported_languages": 45,
            "model_size": "~1.2GB",
            "loaded": self._model is not None
        }


# Singleton instance
_xlsum_service: Optional[XLSumService] = None


def get_xlsum_service() -> XLSumService:
    """Get or create XLSumService singleton"""
    global _xlsum_service
    if _xlsum_service is None:
        _xlsum_service = XLSumService()
    return _xlsum_service
