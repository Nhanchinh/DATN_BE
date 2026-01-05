"""
VietNews Summarization Service
Model: Local finetuned ViT5 model (my_vit5_model_finetune10k)
Ideal for: Vietnamese news summarization
"""

import logging
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class VietNewsService:
    """
    Summarization using local finetuned ViT5 model.
    This model is finetuned on 10k samples for Vietnamese news summarization.
    """
    
    MODEL_NAME = "AI_Models/my_vit5_model_finetune10k"
    
    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"VietNewsService initialized. Device: {self._device}")
    
    def _load_model(self) -> None:
        """Lazy load VietNews model"""
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
        max_length: int = 256,
        min_length: int = 40
    ) -> Tuple[str, str]:
        """
        Summarize text using VietNews model.
        
        Args:
            text: Input news text
            max_length: Max summary length
            min_length: Min summary length
            
        Returns:
            Tuple[str, str]: (raw_summary, processed_summary)
        """
        self._load_model()
        
        # VietNews model expects "vietnews: " prefix + text
        # Adding explicit prompt as requested to improve context understanding
        input_text = "vietnews: " + text
        
        inputs = self._tokenizer(
            input_text,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding="max_length"
        ).to(self._device)
        
        with torch.inference_mode():
            outputs = self._model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=150,
                min_length=min_length,
                num_beams=5,               # Mức cân bằng: không chậm như 5, không ngáo như 1-2
                early_stopping=True,
                no_repeat_ngram_size=3,
                repetition_penalty=2.5,    # Giữ mức này để tránh lặp nhưng không làm gãy câu
                length_penalty=1.2         # Khuyến khích model nối ý nhưng không quá dài
            )
        
        summary = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Simple processing
        processed_summary = summary.replace(" .", ".").strip()
        
        return summary, processed_summary
    
    def get_model_info(self) -> dict:
        return {
            "model_name": self.MODEL_NAME,
            "description": "Local finetuned ViT5 model trained on 10k samples for Vietnamese news summarization",
            "type": "Abstractive Summarization",
            "domain": "News",
            "model_size": "~900MB",
            "loaded": self._model is not None
        }


# Singleton
_vietnews_service: Optional[VietNewsService] = None


def get_vietnews_service() -> VietNewsService:
    global _vietnews_service
    if _vietnews_service is None:
        _vietnews_service = VietNewsService()
    return _vietnews_service
