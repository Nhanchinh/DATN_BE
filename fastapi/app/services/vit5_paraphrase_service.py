"""
ViT5 Paraphrase Service - Smooth & Refine Text
Model: AI_Models/myvitparaphrase
Prefix: "làm mượt:" (Make smooth)

This service takes extracted sentences from PhoBERT and paraphrases them
into smooth, natural Vietnamese text with proper connectors.
"""

import logging
from typing import List, Tuple

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logger = logging.getLogger(__name__)


class ViT5ParaphraseService:
    """
    ViT5 Paraphrase - Smooth extracted sentences into natural text.
    
    KEY FEATURES:
    1. Chunking: Process 3 sentences at a time (prevent 256 token overflow)
    2. Paraphrasing: Add connectors, fix grammar, make text flow
    3. Prefix: "làm mượt:" (trained with this prefix)
    
    PIPELINE:
    PhoBERT extract → Sort by index → Chunk (3 sentences) → ViT5 Paraphrase → Join
    """
    
    MODEL_NAME = "AI_Models/my_vit5_paraphrase_model"
    PREFIX = "làm mượt: "  # CRITICAL: Model trained with this prefix
    
    # PARAPHRASE CONFIG (Optimized for smooth text)
    GEN_CONFIG = {
        "max_length": 256,           # Allow long paraphrases
        "min_length": 20,            # Sufficient for meaningful text
        "num_beams": 5,              # High quality - think about connections
        "repetition_penalty": 1.1,   # Light penalty
        "early_stopping": True,
    }
    
    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"ViT5ParaphraseService initialized. Device: {self._device}")
    
    def _load_model(self) -> None:
        """Lazy load ViT5 Paraphrase model"""
        if self._model is None:
            logger.info(f"Loading {self.MODEL_NAME}... (first time may take 1-2 min)")
            self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.MODEL_NAME)
            self._model.to(self._device)
            self._model.eval()
            logger.info(f"{self.MODEL_NAME} loaded successfully!")
    
    def _post_process_text(self, text: str) -> str:
        """
        Post-process paraphrased text to fix model's inherent bugs.
        
        KNOWN ISSUES TO FIX:
        1. "COVID-19" → "Saxifrag-19" (tokenizer bug)
        2. Number spacing issues
        
        Args:
            text: Raw paraphrased text from model
            
        Returns:
            Cleaned text
        """
        import re
        
        # Fix COVID-19 tokenizer bug (CRITICAL!)
        text = text.replace("Saxifrag-19", "COVID-19")
        text = text.replace("Saxifrag", "COVID")
        
        # Fix number spacing (e.g., "3, 32%" -> "3,32%")
        text = re.sub(r'(\d+),\s+(\d+)', r'\1,\2', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def paraphrase_chunk(
        self,
        text: str,
        max_length: int = 256,
        min_length: int = 20
    ) -> str:
        """
        Paraphrase a chunk of text (typically 3 sentences).
        
        Args:
            text: Input text to paraphrase
            max_length: Max output length
            min_length: Min output length
            
        Returns:
            Smooth, paraphrased text
        """
        self._load_model()
        
        # Add prefix
        input_text = self.PREFIX + text.strip()
        
        # Tokenize
        inputs = self._tokenizer(
            input_text,
            return_tensors="pt",
            max_length=256,
            truncation=True,
            padding="max_length"
        ).to(self._device)
        
        # Override config
        gen_config = self.GEN_CONFIG.copy()
        gen_config["max_length"] = max_length
        gen_config["min_length"] = min_length
        
        # Generate
        with torch.no_grad():
            output_ids = self._model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **gen_config
            )
        
        # Decode
        paraphrased = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # POST-PROCESSING: Fix model's inherent bugs
        paraphrased = self._post_process_text(paraphrased)
        
        return paraphrased.strip()
    
    def paraphrase_sentences(
        self,
        sentences: List[str],
        chunk_size: int = 3,
        max_length: int = 256,
        min_length: int = 20
    ) -> str:
        """
        Paraphrase a list of sentences with chunking strategy.
        
        STRATEGY:
        1. Group sentences into chunks of `chunk_size` (default 3)
        2. Paraphrase each chunk separately
        3. Join all paraphrased chunks
        
        Args:
            sentences: List of sentences to paraphrase
            chunk_size: Number of sentences per chunk (default 3)
            max_length: Max length for each chunk output
            min_length: Min length for each chunk output
            
        Returns:
            Final smooth text (all chunks joined)
        """
        self._load_model()
        
        paraphrased_parts = []
        
        # Process in chunks
        for i in range(0, len(sentences), chunk_size):
            # Get chunk (3 sentences)
            chunk_sentences = sentences[i:i + chunk_size]
            
            # Join raw (input for paraphrase)
            raw_input = " ".join(chunk_sentences)
            
            # Paraphrase this chunk
            smooth_chunk = self.paraphrase_chunk(
                text=raw_input,
                max_length=max_length,
                min_length=min_length
            )
            
            paraphrased_parts.append(smooth_chunk)
        
        # Join all paraphrased chunks
        final_text = " ".join(paraphrased_parts)
        
        return final_text
    
    def get_model_info(self) -> dict:
        """Return model information"""
        return {
            "model_name": self.MODEL_NAME,
            "prefix": self.PREFIX,
            "description": "ViT5 finetuned for paraphrasing with 'làm mượt:' prefix",
            "approach": "Chunking (3 sentences) + Paraphrasing",
            "config": self.GEN_CONFIG,
            "supported_languages": ["vi"],
            "model_size": "~900MB",
            "loaded": self._model is not None,
            "advantages": [
                "Add connectors between sentences",
                "Fix grammar and make text flow naturally",
                "Chunking prevents 256 token overflow"
            ]
        }


# Singleton instance
_vit5_paraphrase_service = None


def get_vit5_paraphrase_service() -> ViT5ParaphraseService:
    """Get or create ViT5ParaphraseService singleton"""
    global _vit5_paraphrase_service
    if _vit5_paraphrase_service is None:
        _vit5_paraphrase_service = ViT5ParaphraseService()
    return _vit5_paraphrase_service
