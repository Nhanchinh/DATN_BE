"""
Text Processing Utilities for Summarization
Pre-processing and Post-processing to improve summary quality
"""

import re
from typing import Dict, List, Set, Tuple


class TextProcessor:
    """
    Pre-processing and Post-processing utilities for text summarization.
    
    Pre-processing:
    - Clean and normalize text
    - Extract key entities/topics
    - Identify sections for balanced summarization
    
    Post-processing:
    - Remove redundant phrases
    - Remove duplicate words in close proximity
    - Ensure key entities are preserved
    """
    
    def __init__(self):
        # Common redundant patterns to remove
        self.redundant_patterns = [
            # "X's Y ... developed by X" -> remove second X
            (r"(\b\w+)'s\s+(\w+.*?)developed by \1", r"\1's \2"),
            # Double mentions like "Microsoft... Microsoft"
            (r"(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b(.*?)\b\1\b", self._remove_second_mention),
        ]
    
    def preprocess(self, text: str) -> str:
        """
        Pre-process text before summarization.
        
        Steps:
        1. Normalize whitespace
        2. Fix punctuation
        3. Ensure proper sentence structure
        """
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Ensure single space after punctuation
        text = re.sub(r'([.!?])\s*', r'\1 ', text)
        
        # Remove multiple periods
        text = re.sub(r'\.{2,}', '.', text)
        
        # Ensure text ends with proper punctuation
        if text and text[-1] not in '.!?':
            text += '.'
        
        return text
    
    def extract_entities(self, text: str) -> List[str]:
        """
        Extract important named entities from text.
        
        Looks for:
        - Capitalized words/phrases (product names, companies)
        - Technical terms
        """
        entities = []
        
        # Pattern for capitalized terms (likely proper nouns)
        # Matches: "Android Studio", "Visual Studio Code", "Microsoft", etc.
        pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+[A-Z]+)?)\b'
        matches = re.findall(pattern, text)
        
        # Filter out common words that might be capitalized at sentence start
        common_words = {'The', 'This', 'It', 'They', 'These', 'Those', 'In', 'On', 'With', 'For', 'And', 'But', 'Or'}
        entities = [m for m in matches if m not in common_words and len(m) > 2]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_entities = []
        for e in entities:
            if e.lower() not in seen:
                seen.add(e.lower())
                unique_entities.append(e)
        
        return unique_entities
    
    def extract_topics(self, text: str) -> List[Dict[str, str]]:
        """
        Extract distinct MAJOR topics/sections from text.
        
        A new topic is detected when:
        1. A multi-word proper noun appears at sentence start (e.g., "Android Studio", "Visual Studio Code")
        2. A transition phrase is used (e.g., "On the other hand", "In contrast")
        3. A definition pattern "X is/are ..." introduces a new subject
        
        Returns list of dicts with:
        - name: Topic name (main entity)
        - content: Text content for that topic
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Transition phrases that indicate topic change
        transition_phrases = [
            'on the other hand', 'in contrast', 'however', 'meanwhile',
            'alternatively', 'conversely', 'whereas', 'while'
        ]
        
        # Pattern for multi-word product/company names at sentence start
        # Must be 2+ words with capitals: "Android Studio", "Visual Studio Code"
        multiword_entity_pattern = r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)'
        
        topics = []
        current_topic = {'name': '', 'content': '', 'entities': []}
        
        for sentence in sentences:
            sentence_lower = sentence.lower().strip()
            
            # Check for transition phrases
            has_transition = any(phrase in sentence_lower for phrase in transition_phrases)
            
            # Check for multi-word entity at start of sentence
            multiword_match = re.match(multiword_entity_pattern, sentence.strip())
            
            # Check if sentence is a definition: "X is/are the/a ..."
            is_definition = bool(re.match(r'^[A-Z][a-z]+.*?\s+(is|are)\s+(the|a|an)\s+', sentence.strip()))
            
            # Decide if this starts a new topic
            should_start_new_topic = False
            new_topic_name = ''
            
            if multiword_match:
                candidate_name = multiword_match.group(1)
                # Only start new topic if this is different from current topic
                if candidate_name.lower() != current_topic['name'].lower():
                    should_start_new_topic = True
                    new_topic_name = candidate_name
            
            if has_transition and not should_start_new_topic:
                # Look for entity after transition
                entities = self.extract_entities(sentence)
                if entities and entities[0].lower() != current_topic['name'].lower():
                    should_start_new_topic = True
                    new_topic_name = entities[0] if len(entities[0].split()) > 1 else ''
            
            if should_start_new_topic and new_topic_name:
                # Save current topic if it has content
                if current_topic['content']:
                    topics.append(current_topic)
                
                # Start new topic
                current_topic = {
                    'name': new_topic_name,
                    'content': sentence,
                    'entities': self.extract_entities(sentence)
                }
            else:
                # Add to current topic
                if current_topic['content']:
                    current_topic['content'] += ' ' + sentence
                else:
                    # First sentence - try to find a good topic name
                    entities = self.extract_entities(sentence)
                    multiword = [e for e in entities if len(e.split()) > 1]
                    current_topic['name'] = multiword[0] if multiword else (entities[0] if entities else '')
                    current_topic['content'] = sentence
                current_topic['entities'].extend(self.extract_entities(sentence))
        
        # Don't forget the last topic
        if current_topic['content']:
            topics.append(current_topic)
        
        return topics
    
    def _remove_second_mention(self, match: re.Match) -> str:
        """Helper to remove second mention of an entity in close proximity."""
        entity = match.group(1)
        middle = match.group(2)
        # Keep first mention, remove second
        return f"{entity}{middle}"
    
    def postprocess(self, summary: str, original_entities: List[str] = None) -> str:
        """
        Post-process summary to improve quality.
        
        Steps:
        1. Remove specific redundant patterns (e.g., "Microsoft's... developed by Microsoft")
        2. Fix spacing and punctuation
        
        NOTE: We do NOT remove individual duplicate words anymore because
        this can break entity names (e.g., "Visual" in "Visual Studio Code"
        was being removed due to "Studio" also appearing in "Android Studio").
        """
        result = summary
        
        # Step 1: Remove redundant patterns like "Microsoft's... developed by Microsoft"
        # This is the specific pattern that causes "lặp từ" issues
        result = re.sub(
            r"(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'s\s+(.*?)\s+(?:developed|created|made|built)\s+by\s+\1\b",
            r"\1's \2",
            result
        )
        
        # Step 2: Remove exact duplicate sentences (if any)
        sentences = result.split('. ')
        seen_sentences = set()
        unique_sentences = []
        for s in sentences:
            s_normalized = s.lower().strip()
            if s_normalized and s_normalized not in seen_sentences:
                seen_sentences.add(s_normalized)
                unique_sentences.append(s)
        result = '. '.join(unique_sentences)
        
        # Step 3: Fix spacing and punctuation
        result = re.sub(r'\s+', ' ', result).strip()
        result = re.sub(r'\s+([.,!?])', r'\1', result)
        
        # Step 4: Ensure proper ending
        if result and result[-1] not in '.!?':
            result += '.'
        
        return result
    
    def check_entity_coverage(self, summary: str, original_text: str) -> Dict[str, bool]:
        """
        Check if important entities from original text appear in summary.
        
        Returns dict mapping entity -> whether it appears in summary
        """
        original_entities = self.extract_entities(original_text)
        summary_lower = summary.lower()
        
        coverage = {}
        for entity in original_entities[:10]:  # Check top 10 entities
            coverage[entity] = entity.lower() in summary_lower
        
        return coverage
    
    def calculate_topic_balance(self, summary: str, topics: List[Dict]) -> Dict[str, float]:
        """
        Calculate how well balanced the summary covers each topic.
        
        Returns dict mapping topic name -> coverage ratio (0-1)
        """
        if not topics:
            return {}
        
        summary_lower = summary.lower()
        balance = {}
        
        for topic in topics:
            if not topic['name']:
                continue
            
            # Count how many of the topic's entities appear in summary
            entities = topic.get('entities', [])
            if not entities:
                continue
            
            covered = sum(1 for e in entities if e.lower() in summary_lower)
            balance[topic['name']] = round(covered / len(entities), 2) if entities else 0
        
        return balance


# Singleton instance
_text_processor = None


def get_text_processor() -> TextProcessor:
    """Get or create TextProcessor singleton"""
    global _text_processor
    if _text_processor is None:
        _text_processor = TextProcessor()
    return _text_processor
