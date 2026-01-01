"""
Qwen Service - Lightweight LLM for Text Fusion/Rewriting
Uses Qwen2.5-0.5B-Instruct for intelligent text processing
"""

import logging
from typing import Optional, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.utils.text_processor import TextProcessor, get_text_processor

logger = logging.getLogger(__name__)


class QwenService:
    """
    Lightweight LLM service using Qwen2.5-0.5B-Instruct.
    
    Qwen2.5-0.5B is an extremely lightweight model (~500MB) that can:
    - Fuse fragmented sentences into fluent paragraphs
    - Paraphrase text with natural style
    - Follow instructions for text processing
    
    Much smarter than BARTpho for understanding instructions,
    while still being lightweight enough to run on CPU.
    """
    
    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
    
    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._text_processor: TextProcessor = get_text_processor()
        
        logger.info(f"Qwen Service initialized. Device: {self._device}")
    
    def _load_model(self) -> None:
        """Lazy load Qwen model"""
        if self._model is None:
            logger.info(f"Loading {self.MODEL_NAME}... (this may take 1-2 minutes)")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.MODEL_NAME,
                trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.MODEL_NAME,
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
                trust_remote_code=True
            )
            self._model.to(self._device)
            self._model.eval()
            logger.info(f"{self.MODEL_NAME} loaded successfully!")
    
    def _generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        """Generate text from prompt"""
        self._load_model()
        
        messages = [
            {"role": "system", "content": "Bạn là trợ lý AI chuyên viết lại văn bản tiếng Việt. Hãy viết ngắn gọn, mạch lạc, giữ nguyên thông tin quan trọng."},
            {"role": "user", "content": prompt}
        ]
        
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self._tokenizer([text], return_tensors="pt").to(self._device)
        
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.3,  # Low temperature for more focused output
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self._tokenizer.eos_token_id
            )
        
        # Decode only the new tokens
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        response = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return response.strip()
    
    def fuse_sentences(
        self,
        sentences: List[str],
        max_new_tokens: int = 256
    ) -> str:
        """
        Fuse multiple fragmented sentences into a fluent paragraph.
        
        Args:
            sentences: List of extracted sentences to fuse
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            str: A fluent, coherent paragraph
        """
        sentences_text = "\n".join([f"- {s}" for s in sentences])
        
        prompt = f"""Dựa vào các ý chính sau, hãy viết lại thành MỘT đoạn văn ngắn gọn, mạch lạc bằng tiếng Việt.
Chỉ sử dụng thông tin đã cho, KHÔNG thêm thông tin mới.

Các ý chính:
{sentences_text}

Đoạn văn viết lại:"""
        
        result = self._generate(prompt, max_new_tokens)
        
        # Clean up
        result = result.strip()
        if result and result[-1] not in '.!?':
            result += '.'
        
        return result
    
    def paraphrase(
        self,
        text: str,
        max_new_tokens: int = 256
    ) -> str:
        """
        Paraphrase/rewrite text while preserving meaning.
        
        Args:
            text: Input text to paraphrase
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            str: Paraphrased text with improved fluency
        """
        prompt = f"""Viết lại đoạn văn sau thành một đoạn văn mượt mà, trôi chảy hơn.
Giữ nguyên tất cả thông tin và số liệu quan trọng.

Đoạn văn gốc:
{text}

Đoạn văn viết lại:"""
        
        result = self._generate(prompt, max_new_tokens)
        
        result = result.strip()
        if result and result[-1] not in '.!?':
            result += '.'
        
        return result
    
    def summarize(
        self,
        text: str,
        max_new_tokens: int = 150
    ) -> str:
        """
        Summarize text using Qwen.
        
        Args:
            text: Input text to summarize
            max_new_tokens: Maximum tokens for summary
            
        Returns:
            str: Summary of the input text
        """
        prompt = f"""Tóm tắt đoạn văn sau thành 2-3 câu ngắn gọn, giữ lại các thông tin và số liệu quan trọng nhất.

Đoạn văn:
{text}

Tóm tắt:"""
        
        result = self._generate(prompt, max_new_tokens)
        
        result = result.strip()
        if result and result[-1] not in '.!?':
            result += '.'
        
        return result
    
    def get_model_info(self) -> dict:
        """Return information about the model"""
        return {
            "model_name": self.MODEL_NAME,
            "description": "Qwen2.5-0.5B - Lightweight instruction-following LLM by Alibaba",
            "capabilities": [
                "Text Fusion (nối câu rời rạc thành đoạn văn)",
                "Paraphrasing (viết lại văn bản mượt mà)",
                "Summarization (tóm tắt có hướng dẫn)",
                "Instruction Following (hiểu và làm theo yêu cầu)"
            ],
            "supported_languages": ["vi", "en", "zh", "và 20+ ngôn ngữ khác"],
            "model_size": "~500MB",
            "loaded": self._model is not None,
            "organization": "Alibaba Cloud",
            "advantages": [
                "Extremely lightweight (chạy được trên CPU)",
                "Smart instruction following",
                "Low hallucination with proper prompting",
                "Natural Vietnamese writing style"
            ]
        }


# Singleton instance for dependency injection
_qwen_service: Optional[QwenService] = None


def get_qwen_service() -> QwenService:
    """Get or create QwenService singleton"""
    global _qwen_service
    if _qwen_service is None:
        _qwen_service = QwenService()
    return _qwen_service
