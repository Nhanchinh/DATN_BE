"""
Summarization Service - Using BART-Large-CNN (Summarization Specialist)

This model is specifically fine-tuned on 300K CNN/DailyMail news articles
for abstractive summarization. It produces high-quality summaries that:
- Preserve proper nouns and key entities
- Generate fluent, natural language
- Capture the main ideas accurately
"""

import logging
from typing import Optional, Tuple

import torch
from transformers import BartForConditionalGeneration, BartTokenizer

logger = logging.getLogger(__name__)


class SummarizationService:
    """
    Summarization service using facebook/bart-large-cnn
    
    This is a single-model approach using a specialist model that's
    already optimized for summarization tasks.
    """
    
    MODEL_NAME = "facebook/bart-large-cnn"
    
    def __init__(self):
        self._model: Optional[BartForConditionalGeneration] = None
        self._tokenizer: Optional[BartTokenizer] = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"SummarizationService initialized. Device: {self._device}")
    
    def _load_model(self) -> None:
        """Lazy load BART-large-cnn model"""
        if self._model is None:
            logger.info(f"Loading {self.MODEL_NAME}... (this may take a minute)")
            self._tokenizer = BartTokenizer.from_pretrained(self.MODEL_NAME)
            self._model = BartForConditionalGeneration.from_pretrained(self.MODEL_NAME)
            self._model.to(self._device)
            self._model.eval()
            logger.info(f"{self.MODEL_NAME} loaded successfully!")
    
    def generate_raw_summary(
        self,
        text: str,
        max_length: int = 150,
        min_length: int = 30
    ) -> str:
        """
        Generate summary using BART-large-cnn
        
        This is "Stage 1" - the initial summarization.
        Since BART-large-cnn is already excellent at summarization,
        this produces high-quality output directly.
        """
        self._load_model()
        
        # BART-large-cnn doesn't need special prompts - just the text
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            max_length=1024,  # BART can handle longer inputs
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
        
        summary = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary
    
    def refine_summary(
        self,
        text: str,
        max_length: int = 150
    ) -> str:
        """
        Stage 2: Further refine/condense the summary
        
        For BART-large-cnn, we can run it again on the summary
        to potentially make it more concise or fluent.
        """
        self._load_model()
        
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self._device)
        
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_length=max_length,
                min_length=20,
                num_beams=4,
                length_penalty=1.5,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
        
        refined = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        return refined
    
    def summarize(
        self,
        text: str,
        max_length: int = 150,
        min_length: int = 30
    ) -> Tuple[str, str]:
        """
        Full summarization pipeline:
        1. Generate initial summary from original text
        2. Optionally refine (for two-stage comparison)
        
        Returns:
            Tuple[str, str]: (raw_summary, final_summary)
        """
        # Stage 1: Initial summarization
        raw_summary = self.generate_raw_summary(text, max_length, min_length)
        
        # Stage 2: For comparison, we'll keep raw and refined the same
        # since BART-large-cnn already produces excellent summaries
        # Running it twice on short text doesn't improve much
        final_summary = raw_summary
        
        return raw_summary, final_summary
    
    def calculate_improvement_ratio(self, original: str, refined: str) -> float:
        """Calculate how much the text changed after refinement"""
        if not original or not refined:
            return 0.0
        
        original_words = set(original.lower().split())
        refined_words = set(refined.lower().split())
        
        if not original_words:
            return 1.0
        
        intersection = len(original_words & refined_words)
        union = len(original_words | refined_words)
        
        similarity = intersection / union if union > 0 else 0
        return round(1 - similarity, 3)


# Singleton instance for dependency injection
_summarization_service: Optional[SummarizationService] = None


def get_summarization_service() -> SummarizationService:
    """Get or create SummarizationService singleton"""
    global _summarization_service
    if _summarization_service is None:
        _summarization_service = SummarizationService()
    return _summarization_service
