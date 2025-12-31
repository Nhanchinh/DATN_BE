"""
Extractive Summarization Service - Custom Implementation
Safe summarization that extracts important sentences without hallucination
No external dependency issues!
"""

import logging
import re
from typing import List, Optional, Tuple

import torch
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class ExtractiveSummarizationService:
    """
    Extractive summarization service using PhoBERT with custom implementation.
    
    Unlike abstractive models (T5, BART, ViT5), extractive summarization:
    - NEVER hallucinates or makes up information
    - Only selects and returns original sentences from the text
    - 100% faithful to the source content
    
    Uses PhoBERT (Vietnamese BERT) to compute sentence embeddings
    and select the most representative sentences.
    """
    
    MODEL_NAME = "vinai/phobert-base"
    
    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"ExtractiveSummarizationService initialized. Device: {self._device}")
    
    def _load_model(self) -> None:
        """Lazy load PhoBERT model"""
        if self._model is None:
            logger.info(f"Loading {self.MODEL_NAME}... (this may take a minute)")
            
            self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
            self._model = AutoModel.from_pretrained(self.MODEL_NAME)
            self._model.to(self._device)
            self._model.eval()
            
            logger.info(f"{self.MODEL_NAME} loaded successfully!")
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences for Vietnamese"""
        # Split on Vietnamese sentence endings
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    def _get_sentence_embedding(self, sentence: str) -> torch.Tensor:
        """Get embedding for a single sentence using PhoBERT"""
        inputs = self._tokenizer(
            sentence,
            return_tensors="pt",
            max_length=256,
            truncation=True,
            padding=True
        ).to(self._device)
        
        with torch.no_grad():
            outputs = self._model(**inputs)
            # Use mean pooling of last hidden state
            embedding = outputs.last_hidden_state.mean(dim=1)
        
        return embedding.squeeze()
    
    def _compute_sentence_scores(self, sentences: List[str]) -> List[float]:
        """
        Compute importance scores for each sentence.
        
        Uses cosine similarity between sentence embedding and document embedding.
        Sentences more similar to the overall document are considered more important.
        """
        if not sentences:
            return []
        
        self._load_model()
        
        # Get embeddings for all sentences
        embeddings = []
        for sentence in sentences:
            emb = self._get_sentence_embedding(sentence)
            embeddings.append(emb)
        
        # Stack embeddings
        sentence_embeddings = torch.stack(embeddings)
        
        # Document embedding = mean of all sentence embeddings
        doc_embedding = sentence_embeddings.mean(dim=0)
        
        # Compute cosine similarity between each sentence and document
        scores = []
        for emb in sentence_embeddings:
            similarity = torch.nn.functional.cosine_similarity(
                emb.unsqueeze(0),
                doc_embedding.unsqueeze(0)
            ).item()
            scores.append(similarity)
        
        # Add position bias (first sentences usually more important)
        for i in range(len(scores)):
            position_bonus = 0.1 * (1 - i / len(scores))
            scores[i] += position_bonus
        
        return scores
    
    def summarize(
        self,
        text: str,
        ratio: float = 0.3,
        num_sentences: Optional[int] = None
    ) -> Tuple[str, List[str]]:
        """
        Extract important sentences from text.
        
        Args:
            text: Input text to summarize
            ratio: Percentage of sentences to keep (0.0 to 1.0)
            num_sentences: If specified, extract exactly this many sentences
            
        Returns:
            Tuple[str, List[str]]: (summary as string, list of extracted sentences)
        """
        # Split into sentences
        sentences = self._split_sentences(text)
        
        # If text is too short, just return it
        if len(sentences) <= 2:
            return text, sentences
        
        # Compute importance scores
        scores = self._compute_sentence_scores(sentences)
        
        # Determine number of sentences to extract
        if num_sentences:
            n_extract = min(num_sentences, len(sentences))
        else:
            n_extract = max(1, int(len(sentences) * ratio))
        
        # Get top-scored sentences
        scored_sentences = list(zip(range(len(sentences)), sentences, scores))
        
        # Sort by score (descending)
        scored_sentences.sort(key=lambda x: x[2], reverse=True)
        
        # Take top n sentences
        top_sentences = scored_sentences[:n_extract]
        
        # Sort back by original position for coherent reading
        top_sentences.sort(key=lambda x: x[0])
        
        # Extract just the sentences
        extracted_sentences = [s[1] for s in top_sentences]
        
        # Join into summary
        summary = ' '.join(extracted_sentences)
        
        return summary, extracted_sentences
    
    def summarize_by_sentences(
        self,
        text: str,
        num_sentences: int = 3
    ) -> Tuple[str, List[str]]:
        """
        Extract a specific number of important sentences.
        
        Args:
            text: Input text to summarize
            num_sentences: Number of sentences to extract
            
        Returns:
            Tuple[str, List[str]]: (summary as string, list of extracted sentences)
        """
        return self.summarize(text, num_sentences=num_sentences)
    
    def get_model_info(self) -> dict:
        """Return information about the model"""
        return {
            "model_name": self.MODEL_NAME,
            "method": "Extractive Summarization (Custom)",
            "description": "PhoBERT-based extractive summarizer - selects important sentences",
            "hallucination_risk": "ZERO - only extracts original sentences",
            "supported_languages": ["vi"],
            "model_size": "~400MB",
            "loaded": self._model is not None
        }


# Singleton instance for dependency injection
_extractive_service: Optional[ExtractiveSummarizationService] = None


def get_extractive_service() -> ExtractiveSummarizationService:
    """Get or create ExtractiveSummarizationService singleton"""
    global _extractive_service
    if _extractive_service is None:
        _extractive_service = ExtractiveSummarizationService()
    return _extractive_service
