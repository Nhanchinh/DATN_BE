"""
BARTpho Service - Vietnamese Text Fusion/Rewriting
Uses VinAI's BARTpho model for paraphrasing and text fusion
"""

import logging
from typing import Optional, Tuple, List

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from app.utils.text_processor import TextProcessor, get_text_processor

logger = logging.getLogger(__name__)


class BARTphoService:
    """
    Vietnamese text fusion/rewriting service using VinAI BARTpho.
    
    BARTpho is a Vietnamese BART model trained on large-scale Vietnamese data.
    It excels at:
    - Paraphrasing: Rewriting text while preserving meaning
    - Text fusion: Combining fragmented sentences into fluent paragraphs
    - Summarization: Condensing text with natural Vietnamese style
    
    Use this after PhoBERT extractive to "polish" the extracted sentences.
    """
    
    MODEL_NAME = "vinai/bartpho-syllable"  # Syllable-level tokenization (better for Vietnamese)
    
    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._text_processor: TextProcessor = get_text_processor()
        
        logger.info(f"BARTpho Service initialized. Device: {self._device}")
    
    def _load_model(self) -> None:
        """Lazy load BARTpho model"""
        if self._model is None:
            logger.info(f"Loading {self.MODEL_NAME}... (this may take 1-2 minutes)")
            self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.MODEL_NAME)
            self._model.to(self._device)
            self._model.eval()
            logger.info(f"{self.MODEL_NAME} loaded successfully!")
    
    def fuse_sentences(
        self,
        sentences: List[str],
        max_length: int = 256,
        min_length: int = 30
    ) -> str:
        """
        Fuse multiple fragmented sentences into a fluent paragraph.
        
        Args:
            sentences: List of extracted sentences to fuse
            max_length: Maximum output length (in tokens)
            min_length: Minimum output length (in tokens)
            
        Returns:
            str: A fluent, coherent paragraph combining all input sentences
        """
        self._load_model()
        
        # Join sentences with space
        input_text = " ".join(sentences)
        
        # Clean the input
        cleaned_text = self._text_processor.preprocess(input_text)
        
        inputs = self._tokenizer(
            cleaned_text,
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
                length_penalty=1.5,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
        
        fused_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Post-process
        fused_text = fused_text.strip()
        if fused_text and fused_text[-1] not in '.!?':
            fused_text += '.'
        
        return fused_text
    
    def paraphrase(
        self,
        text: str,
        max_length: int = 256,
        min_length: int = 20
    ) -> str:
        """
        Paraphrase/rewrite text while preserving meaning.
        
        Args:
            text: Input text to paraphrase
            max_length: Maximum output length
            min_length: Minimum output length
            
        Returns:
            str: Paraphrased text with improved fluency
        """
        self._load_model()
        
        cleaned_text = self._text_processor.preprocess(text)
        
        inputs = self._tokenizer(
            cleaned_text,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        ).to(self._device)
        
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_length=max_length,
                min_length=min_length,
                num_beams=5,  # More beams for better paraphrasing
                length_penalty=1.0,
                early_stopping=True,
                no_repeat_ngram_size=3,
                do_sample=True,  # Enable sampling for more diverse output
                top_p=0.92,
                temperature=0.7,  # Slightly creative but controlled
            )
        
        paraphrased = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        paraphrased = paraphrased.strip()
        if paraphrased and paraphrased[-1] not in '.!?':
            paraphrased += '.'
        
        return paraphrased
    
    def summarize(
        self,
        text: str,
        max_length: int = 150,
        min_length: int = 30
    ) -> Tuple[str, str]:
        """
        Summarize Vietnamese text using BARTpho.
        
        Args:
            text: Input Vietnamese text
            max_length: Maximum summary length
            min_length: Minimum summary length
            
        Returns:
            Tuple[str, str]: (raw_summary, processed_summary)
        """
        self._load_model()
        
        cleaned_text = self._text_processor.preprocess(text)
        
        inputs = self._tokenizer(
            cleaned_text,
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
        
        # Post-processing
        processed_summary = raw_summary.strip()
        if processed_summary and processed_summary[-1] not in '.!?':
            processed_summary += '.'
        
        return raw_summary, processed_summary
    
    def get_model_info(self) -> dict:
        """Return information about the model"""
        return {
            "model_name": self.MODEL_NAME,
            "description": "BARTpho - Vietnamese BART by VinAI for text generation",
            "capabilities": [
                "Paraphrasing (viết lại văn bản)",
                "Text Fusion (nối câu rời rạc)",
                "Summarization (tóm tắt)",
                "Grammar correction (sửa ngữ pháp)"
            ],
            "supported_languages": ["vi"],
            "model_size": "~800MB",
            "loaded": self._model is not None,
            "organization": "VinAI Research",
            "advantages": [
                "Same ecosystem as PhoBERT",
                "Natural Vietnamese writing style",
                "Lower hallucination than LLMs"
            ]
        }


# Singleton instance for dependency injection
_bartpho_service: Optional[BARTphoService] = None


def get_bartpho_service() -> BARTphoService:
    """Get or create BARTphoService singleton"""
    global _bartpho_service
    if _bartpho_service is None:
        _bartpho_service = BARTphoService()
    return _bartpho_service
