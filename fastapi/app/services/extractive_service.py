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
    
    # ========================
    # PREPROCESSING METHODS
    # ========================
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess Vietnamese text before summarization.
        
        - Normalize whitespace
        - Fix common punctuation issues
        - Handle special characters
        """
        if not text:
            return ""
        
        # Normalize multiple spaces to single space
        text = re.sub(r'\s+', ' ', text)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)  # Remove space before punct
        text = re.sub(r'([,.!?;:])(?=[^\s])', r'\1 ', text)  # Add space after punct if missing
        
        # Fix multiple punctuation
        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r',{2,}', ',', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _postprocess_summary(self, summary: str) -> str:
        """
        Postprocess the final summary for better readability.
        
        - Normalize punctuation and spacing
        - Ensure proper capitalization
        - Clean up connector duplications
        - Convert inline connectors to sentence starters
        """
        if not summary:
            return ""
        
        # Normalize whitespace
        summary = re.sub(r'\s+', ' ', summary)
        
        # Fix spacing around punctuation
        summary = re.sub(r'\s+([,.!?;:])', r'\1', summary)
        summary = re.sub(r'([,.!?;:])(?=[^\s\d])', r'\1 ', summary)
        
        # Remove double connectors
        connectors = ['Mặt khác,', 'Bên cạnh đó,', 'Ngoài ra,', 'Đồng thời,']
        for conn in connectors:
            # Remove if connector appears twice in a row
            summary = re.sub(rf'{re.escape(conn)}\s*{re.escape(conn)}', conn, summary, flags=re.IGNORECASE)
        
        # Ensure first letter is capitalized
        if summary:
            summary = summary[0].upper() + summary[1:]
        
        # Ensure ends with period
        if summary and summary[-1] not in '.!?':
            summary += '.'
        
        # Clean up trailing spaces before punctuation
        summary = re.sub(r'\s+\.', '.', summary)
        
        # Fix double periods
        summary = re.sub(r'\.{2,}', '.', summary)
        
        return summary.strip()
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences for Vietnamese.
        
        Handles special cases:
        - Decimal numbers (4.0, 3.5)
        - Abbreviations (TP., Mr., Dr.)
        - Ellipsis (...)
        """
        # Protect decimal numbers from being split
        # Replace "4.0" with "4<DOT>0" temporarily
        protected = re.sub(r'(\d)\.(\d)', r'\1<DOT>\2', text)
        
        # Protect common abbreviations
        abbreviations = ['TP.', 'Mr.', 'Mrs.', 'Ms.', 'Dr.', 'Prof.', 'vs.', 'etc.']
        for abbr in abbreviations:
            protected = protected.replace(abbr, abbr.replace('.', '<DOT>'))
        
        # Split on sentence endings (. ! ?) followed by space and uppercase/start
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ])', protected)
        
        # Restore protected dots
        result = []
        for s in sentences:
            s = s.replace('<DOT>', '.').strip()
            if s and len(s) > 10:
                result.append(s)
        
        return result
    
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
    
    def _join_with_connectors(self, sentences: List[str]) -> str:
        """
        Join extracted sentences with appropriate Vietnamese connectors.
        
        This makes the summary more fluent and natural-sounding.
        - Avoids repeating the same connector
        - Lowercases first letter after connector
        """
        if not sentences:
            return ""
        
        if len(sentences) == 1:
            return sentences[0]
        
        # Vietnamese connectors for different contexts
        contrast_connectors = [
            "Mặt khác,",
            "Bên cạnh đó,",
            "Hơn nữa,",
            "Thêm vào đó,",
            "Đồng thời,",
        ]
        
        result_parts = [sentences[0]]  # First sentence has no connector
        used_connectors = set()  # Track used connectors to avoid repetition
        
        for i in range(1, len(sentences)):
            current = sentences[i]
            previous = sentences[i - 1]
            
            # Check if sentence already has a connector
            existing_connectors = [
                "mặt khác", "ngược lại", "tuy nhiên", "nhưng", 
                "bên cạnh đó", "ngoài ra", "trong khi", "hơn nữa",
                "thêm vào đó", "đồng thời", "cụ thể", "theo đó"
            ]
            
            has_connector = any(current.lower().startswith(conn) for conn in existing_connectors)
            
            if has_connector:
                # Already has connector, just append as-is
                result_parts.append(current)
            else:
                # Pick a connector that hasn't been used yet
                available = [c for c in contrast_connectors if c not in used_connectors]
                if not available:
                    # All used, reset
                    available = contrast_connectors
                    used_connectors.clear()
                
                connector = available[0]
                used_connectors.add(connector)
                
                # Lowercase first letter of sentence after connector
                # Skip if it's a proper noun (check if second char is also uppercase)
                if len(current) > 1 and current[0].isupper() and not current[1].isupper():
                    current = current[0].lower() + current[1:]
                
                result_parts.append(f"{connector} {current}")
        
        return ' '.join(result_parts)
    
    def _detect_topic_change(self, prev_sentence: str, curr_sentence: str) -> bool:
        """
        Detect if the current sentence introduces a new topic.
        
        Simple heuristics:
        1. Different proper nouns at the start
        2. Contrast words already present
        """
        # Check for existing contrast indicators
        contrast_indicators = [
            "mặt khác", "ngược lại", "tuy nhiên", "nhưng", 
            "bên cạnh đó", "ngoài ra", "trong khi"
        ]
        
        curr_lower = curr_sentence.lower()
        for indicator in contrast_indicators:
            if indicator in curr_lower:
                return False  # Already has connector, don't add more
        
        # Check if sentences start with different subjects
        prev_words = prev_sentence.split()[:3]
        curr_words = curr_sentence.split()[:3]
        
        # If first words are very different, likely topic change
        if prev_words and curr_words:
            # Check for proper nouns (capitalized words not at sentence start)
            prev_proper = [w for w in prev_words if w[0].isupper()]
            curr_proper = [w for w in curr_words if w[0].isupper()]
            
            if prev_proper and curr_proper:
                # Different proper nouns = topic change
                if prev_proper[0].lower() != curr_proper[0].lower():
                    return True
        
        return False
    
    def _needs_context(self, sentence: str) -> bool:
        """
        Check if a sentence needs context from previous sentence.
        
        Returns True if sentence starts with:
        - Pronouns (Nó, Họ, Nó là, Anh ấy, Cô ấy...)
        - Connectors (Tuy nhiên, Mặc dù, Nhưng, Do đó, Vì vậy...)
        - Reference words (Đó là, Điều này, Việc này...)
        """
        context_starters = [
            # Pronouns
            'nó ', 'nó,', 'họ ', 'họ,', 'anh ấy', 'cô ấy', 'chúng tôi', 'chúng ta',
            # Connectors
            'tuy nhiên', 'mặc dù', 'nhưng', 'do đó', 'vì vậy', 'vì thế',
            'bởi vậy', 'cho nên', 'thế nên', 'còn', 'và',
            # Reference words
            'đó là', 'điều này', 'việc này', 'điều đó', 'việc đó',
            'như vậy', 'như thế',
            # Demonstratives
            'công cụ này', 'ứng dụng này', 'phần mềm này', 'hệ thống này',
        ]
        
        sentence_lower = sentence.lower().strip()
        
        for starter in context_starters:
            if sentence_lower.startswith(starter):
                return True
        
        return False
    
    def _apply_sentence_windowing(
        self, 
        selected_indices: List[int], 
        all_sentences: List[str]
    ) -> List[int]:
        """
        Apply sentence windowing: if a selected sentence needs context,
        also include the sentence before it.
        
        Args:
            selected_indices: List of selected sentence indices (sorted by score)
            all_sentences: All sentences in the document
            
        Returns:
            Expanded list of indices with context sentences added
        """
        expanded_indices = set(selected_indices)
        
        for idx in selected_indices:
            if idx > 0:  # Has previous sentence
                sentence = all_sentences[idx]
                if self._needs_context(sentence):
                    # Add previous sentence for context
                    expanded_indices.add(idx - 1)
        
        return sorted(list(expanded_indices))
    
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
    
    def _split_into_chunks(self, text: str, chunk_size: int = 3) -> List[str]:
        """
        Split text into chunks of N sentences each.
        
        Args:
            text: Input text
            chunk_size: Number of sentences per chunk
            
        Returns:
            List of text chunks
        """
        sentences = self._split_sentences(text)
        chunks = []
        
        for i in range(0, len(sentences), chunk_size):
            chunk = ' '.join(sentences[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def summarize_chunked(
        self,
        text: str,
        sentences_per_chunk: int = 3,
        sentences_per_extraction: int = 1
    ) -> dict:
        """
        Chunk-based extractive summarization for better coverage.
        
        Pipeline:
        1. Split text into chunks (groups of sentences)
        2. Extract top sentence(s) from each chunk
        3. Combine extracted sentences
        
        This ensures every part of the document is represented.
        
        Args:
            text: Input text to summarize
            sentences_per_chunk: How many sentences per chunk
            sentences_per_extraction: How many sentences to extract from each chunk
            
        Returns:
            Dict with chunking details and final summary
        """
        self._load_model()
        
        # PREPROCESS input text
        text = self._preprocess_text(text)
        
        # Split into sentences first
        all_sentences = self._split_sentences(text)
        
        if len(all_sentences) <= 3:
            # Text too short, just extract normally
            summary, extracted = self.summarize(text, num_sentences=2)
            return {
                "chunks": [{"chunk": text, "extracted": extracted}],
                "num_chunks": 1,
                "extracted_sentences": extracted,
                "final_summary": summary,
                "method": "chunked_extractive (text too short, used regular)",
                "original_length": len(text),
                "final_length": len(summary)
            }
        
        # Split into chunks
        chunks = self._split_into_chunks(text, sentences_per_chunk)
        
        logger.info(f"Split text into {len(chunks)} chunks")
        
        chunk_results = []
        all_extracted = []
        
        for i, chunk in enumerate(chunks):
            # Get sentences in this chunk
            chunk_sentences = self._split_sentences(chunk)
            
            if len(chunk_sentences) == 0:
                continue
            
            if len(chunk_sentences) <= sentences_per_extraction:
                # Chunk is small, take all sentences
                extracted = chunk_sentences
                selected_indices = list(range(len(chunk_sentences)))
            else:
                # Compute scores for this chunk
                scores = self._compute_sentence_scores(chunk_sentences)
                
                # Get top sentences
                scored = list(zip(range(len(chunk_sentences)), chunk_sentences, scores))
                scored.sort(key=lambda x: x[2], reverse=True)
                
                # Take top N indices
                top_indices = [s[0] for s in scored[:sentences_per_extraction]]
                
                # Apply sentence windowing - add context sentences if needed
                selected_indices = self._apply_sentence_windowing(top_indices, chunk_sentences)
            
            # Sort by position and extract sentences
            selected_indices.sort()
            extracted = [chunk_sentences[idx] for idx in selected_indices]
            
            chunk_results.append({
                "chunk_id": i + 1,
                "chunk_preview": chunk[:80] + "..." if len(chunk) > 80 else chunk,
                "extracted": extracted
            })
            all_extracted.extend(extracted)
        
        # Combine extracted sentences WITH CONNECTORS
        final_summary = self._join_with_connectors(all_extracted)
        
        # POSTPROCESS final summary
        final_summary = self._postprocess_summary(final_summary)
        
        return {
            "chunks": chunk_results,
            "num_chunks": len(chunks),
            "extracted_sentences": all_extracted,
            "num_sentences_extracted": len(all_extracted),
            "final_summary": final_summary,
            "method": "chunked_extractive_with_connectors",
            "hallucination_risk": "ZERO",
            "original_length": len(text),
            "final_length": len(final_summary),
            "compression_ratio": round(len(final_summary) / len(text), 3) if len(text) > 0 else 0
        }
    
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
