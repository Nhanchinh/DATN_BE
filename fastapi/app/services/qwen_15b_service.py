"""
Qwen 1.5B Service - More Accurate LLM for Text Fusion/Rewriting
Uses Qwen2.5-1.5B-Instruct for better reasoning and accuracy
"""

import logging
from typing import Optional, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.utils.text_processor import TextProcessor, get_text_processor

logger = logging.getLogger(__name__)


class Qwen15BService:
    """
    More powerful LLM service using Qwen2.5-1.5B-Instruct.
    
    Qwen2.5-1.5B has 3x more parameters than 0.5B, providing:
    - Better reasoning capabilities (who does what)
    - More accurate information retention
    - Less hallucination
    - Still lightweight enough for most laptops (~2GB RAM)
    """
    
    MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
    
    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._text_processor: TextProcessor = get_text_processor()
        
        logger.info(f"Qwen 1.5B Service initialized. Device: {self._device}")
    
    def _load_model(self) -> None:
        """Lazy load Qwen 1.5B model"""
        if self._model is None:
            logger.info(f"Loading {self.MODEL_NAME}... (this may take 2-3 minutes)")
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
    
    def _generate(self, prompt: str, system_prompt: str, max_new_tokens: int = 256) -> str:
        """Generate text from prompt with custom system prompt"""
        self._load_model()
        
        messages = [
            {"role": "system", "content": system_prompt},
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
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self._tokenizer.eos_token_id
            )
        
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
        Uses better reasoning from 1.5B model.
        """
        sentences_text = "\n".join([f"- {s}" for s in sentences])
        
        system_prompt = "Bạn là trợ lý AI chuyên viết lại văn bản tiếng Việt. Hãy viết ngắn gọn, mạch lạc, giữ nguyên thông tin quan trọng và chủ ngữ của các hành động."
        
        prompt = f"""Dựa vào các ý chính sau, hãy viết lại thành MỘT đoạn văn ngắn gọn, mạch lạc bằng tiếng Việt.
Chỉ sử dụng thông tin đã cho, KHÔNG thêm thông tin mới.

Các ý chính:
{sentences_text}

Đoạn văn viết lại:"""
        
        result = self._generate(prompt, system_prompt, max_new_tokens)
        
        result = result.strip()
        if result and result[-1] not in '.!?':
            result += '.'
        
        return result
    
    def summarize(
        self,
        text: str,
        max_new_tokens: int = 150
    ) -> str:
        """Summarize text using Qwen 1.5B."""
        system_prompt = "Bạn là trợ lý AI chuyên tóm tắt văn bản tiếng Việt. Hãy tóm tắt chính xác, giữ nguyên số liệu và chủ ngữ của các hành động."
        
        prompt = f"""Tóm tắt đoạn văn sau thành 2-3 câu ngắn gọn, giữ lại các thông tin và số liệu quan trọng nhất.

Đoạn văn:
{text}

Tóm tắt:"""
        
        result = self._generate(prompt, system_prompt, max_new_tokens)
        
        result = result.strip()
        if result and result[-1] not in '.!?':
            result += '.'
        
        return result
    
    def get_model_info(self) -> dict:
        """Return information about the model"""
        return {
            "model_name": self.MODEL_NAME,
            "description": "Qwen2.5-1.5B - More powerful LLM with better reasoning",
            "parameters": "1.5 billion",
            "capabilities": [
                "Better logical reasoning (ai làm gì)",
                "More accurate information retention",
                "Text Fusion with proper subject-verb agreement",
                "Lower hallucination rate"
            ],
            "supported_languages": ["vi", "en", "zh", "và 20+ ngôn ngữ khác"],
            "model_size": "~1.5GB - 2GB",
            "loaded": self._model is not None,
            "organization": "Alibaba Cloud",
            "comparison_to_0.5b": "3x more parameters = smarter reasoning"
        }


# Singleton instance for dependency injection
_qwen_15b_service: Optional[Qwen15BService] = None


def get_qwen_15b_service() -> Qwen15BService:
    """Get or create Qwen15BService singleton"""
    global _qwen_15b_service
    if _qwen_15b_service is None:
        _qwen_15b_service = Qwen15BService()
    return _qwen_15b_service
