"""
Summarization Service - Using BART-Large-CNN with Pre/Post Processing

Pipeline:
1. PRE-PROCESSING: Clean text, extract entities
2. BART-LARGE-CNN: Generate summary
3. POST-PROCESSING: Remove redundancy, validate coverage
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
from transformers import BartForConditionalGeneration, BartTokenizer

from app.utils.text_processor import TextProcessor, get_text_processor

logger = logging.getLogger(__name__)


class SummarizationService:
    """
    Summarization service using facebook/bart-large-cnn with
    preprocessing and postprocessing for improved quality.
    
    Pipeline:
    1. Preprocess text (clean, normalize)
    2. Extract entities for coverage checking
    3. Generate summary with BART
    4. Postprocess (remove redundancy)
    5. Validate entity coverage
    """
    
    MODEL_NAME = "facebook/bart-large-cnn"
    
    def __init__(self):
        self._model: Optional[BartForConditionalGeneration] = None
        self._tokenizer: Optional[BartTokenizer] = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._text_processor: TextProcessor = get_text_processor()
        
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
        Includes preprocessing for better input quality.
        """
        self._load_model()
        
        # PRE-PROCESSING: Clean and normalize text
        cleaned_text = self._text_processor.preprocess(text)
        
        # BART-large-cnn doesn't need special prompts - just the text
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
        
        summary = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary
    
    def refine_summary(
        self,
        text: str,
        max_length: int = 150
    ) -> str:
        """
        Stage 2: Further refine/condense the summary
        Includes postprocessing for redundancy removal.
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
        Full summarization pipeline with pre/post processing:
        
        1. PRE-PROCESS: Clean and normalize input text
        2. EXTRACT: Get entities from original for coverage check
        3. SUMMARIZE: Generate summary with BART
        4. POST-PROCESS: Remove redundancy and fix issues
        5. VALIDATE: Check entity coverage
        
        Returns:
            Tuple[str, str]: (raw_summary, processed_summary)
        """
        # PRE-PROCESSING
        cleaned_text = self._text_processor.preprocess(text)
        original_entities = self._text_processor.extract_entities(text)
        
        logger.info(f"Extracted entities: {original_entities[:5]}...")
        
        # Stage 1: Initial summarization
        raw_summary = self.generate_raw_summary(cleaned_text, max_length, min_length)
        
        # POST-PROCESSING: Remove redundancy
        processed_summary = self._text_processor.postprocess(raw_summary, original_entities)
        
        # Log coverage info
        coverage = self._text_processor.check_entity_coverage(processed_summary, text)
        covered = sum(1 for v in coverage.values() if v)
        logger.info(f"Entity coverage: {covered}/{len(coverage)}")
        
        return raw_summary, processed_summary
    
    def summarize_with_details(
        self,
        text: str,
        max_length: int = 150,
        min_length: int = 30
    ) -> Dict:
        """
        Full summarization with detailed analysis.
        
        Returns dict with:
        - raw_summary: Before postprocessing
        - final_summary: After postprocessing
        - entities: Extracted entities
        - entity_coverage: Which entities appear in summary
        - topics: Identified topics
        - topic_balance: Coverage per topic
        """
        # PRE-PROCESSING
        cleaned_text = self._text_processor.preprocess(text)
        entities = self._text_processor.extract_entities(text)
        topics = self._text_processor.extract_topics(text)
        
        # SUMMARIZATION
        raw_summary = self.generate_raw_summary(cleaned_text, max_length, min_length)
        
        # POST-PROCESSING
        final_summary = self._text_processor.postprocess(raw_summary, entities)
        
        # ANALYSIS
        entity_coverage = self._text_processor.check_entity_coverage(final_summary, text)
        topic_balance = self._text_processor.calculate_topic_balance(final_summary, topics)
        
        return {
            'raw_summary': raw_summary,
            'final_summary': final_summary,
            'entities': entities[:10],
            'entity_coverage': entity_coverage,
            'topics': [t['name'] for t in topics],
            'topic_balance': topic_balance,
            'original_length': len(text),
            'raw_summary_length': len(raw_summary),
            'final_summary_length': len(final_summary),
        }
    
    def _summarize_single_topic(
        self,
        text: str,
        max_length: int = 80,
        min_length: int = 10
    ) -> str:
        """Summarize a single topic/section with aggressive compression"""
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
                min_length=min_length,
                num_beams=4,
                length_penalty=3.0,  # Increased from 1.5 to prefer shorter outputs
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
        
        return self._tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def summarize_balanced(
        self,
        text: str,
        max_length: int = 200,
        min_length: int = 50
    ) -> Tuple[str, str, List[Dict]]:
        """
        Topic-balanced summarization:
        1. Split text into topics
        2. Summarize each topic with equal quota
        3. Combine into balanced final summary
        
        This ensures all topics get fair coverage, solving the problem
        where BART focuses too much on early content.
        
        Returns:
            Tuple[str, str, List[Dict]]: (raw_combined, final_summary, topic_summaries)
        """
        # PRE-PROCESSING
        cleaned_text = self._text_processor.preprocess(text)
        topics = self._text_processor.extract_topics(text)
        entities = self._text_processor.extract_entities(text)
        
        logger.info(f"Found {len(topics)} topics: {[t['name'] for t in topics]}")
        
        if len(topics) <= 1:
            # If only 1 topic, use regular summarization
            raw, processed = self.summarize(text, max_length, min_length)
            return raw, processed, []
        
        # Calculate per-topic length quota
        # Give each topic roughly equal space, allowing shorter summaries
        per_topic_max = max(30, max_length // len(topics))  # Reduced from 50
        per_topic_min = max(10, min_length // len(topics))  # Reduced from 15
        
        topic_summaries = []
        
        for topic in topics:
            if not topic['content'] or len(topic['content'].strip()) < 30:
                continue
            
            # Summarize this topic
            topic_summary = self._summarize_single_topic(
                topic['content'],
                max_length=per_topic_max,
                min_length=per_topic_min
            )
            
            topic_summaries.append({
                'name': topic['name'],
                'original': topic['content'][:100] + '...',
                'summary': topic_summary
            })
            
            logger.info(f"Topic '{topic['name']}' summarized: {len(topic_summary)} chars")
        
        # Combine topic summaries
        raw_combined = ' '.join([ts['summary'] for ts in topic_summaries])
        
        # POST-PROCESSING
        final_summary = self._text_processor.postprocess(raw_combined, entities)
        
        return raw_combined, final_summary, topic_summaries
    
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
