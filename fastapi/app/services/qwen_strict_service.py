"""
Qwen Strict Service - Qwen with Strict Prompting for Maximum Accuracy
Uses enhanced prompts to minimize hallucination and maintain factual accuracy
"""

import logging
from typing import Optional, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.utils.text_processor import TextProcessor, get_text_processor

logger = logging.getLogger(__name__)


# Strict system prompt to minimize hallucination
STRICT_SYSTEM_PROMPT = """Bạn là một trợ lý tóm tắt tin tức CHÍNH XÁC. Nhiệm vụ của bạn là viết lại các câu văn đầu vào thành một đoạn văn mượt mà.

QUY TẮC BẮT BUỘC:
1. KHÔNG thêm bất kỳ thông tin nào bên ngoài (như ngày tháng, đánh giá cá nhân, so sánh không có trong bài).
2. GIỮ NGUYÊN chủ ngữ của các hành động (Ai làm gì - không được đổi).
3. GIỮ NGUYÊN tất cả số liệu và con số chính xác.
4. Chỉ dùng từ nối đơn giản để liên kết câu (như "đồng thời", "bên cạnh đó", "ngoài ra").
5. KHÔNG thêm tính từ đánh giá (như "quý giá", "ấn tượng", "đặc biệt").
6. KHÔNG suy diễn thêm thông tin (ví dụ: không thêm "so với năm trước" nếu không có trong gốc).

Hãy viết lại văn bản một cách TRUNG THỰC và CHÍNH XÁC."""


class QwenStrictService:
    """
    Qwen service with STRICT prompting for maximum accuracy.
    
    This service uses the same 0.5B model but with carefully crafted
    prompts to minimize hallucination and maintain factual accuracy.
    
    Trade-off: Less creative/fluent, but more accurate.
    """
    
    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
    
    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._text_processor: TextProcessor = get_text_processor()
        
        logger.info(f"Qwen Strict Service initialized. Device: {self._device}")
    
    def _load_model(self) -> None:
        """Lazy load Qwen model"""
        if self._model is None:
            logger.info(f"Loading {self.MODEL_NAME} (Strict Mode)...")
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
            logger.info(f"{self.MODEL_NAME} (Strict Mode) loaded successfully!")
    
    def _generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        """Generate text with strict prompting"""
        self._load_model()
        
        messages = [
            {"role": "system", "content": STRICT_SYSTEM_PROMPT},
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
                temperature=0.2,  # Lower temperature for more deterministic output
                top_p=0.85,  # More focused sampling
                repetition_penalty=1.15,  # Stronger repetition penalty
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
        Fuse sentences with strict accuracy constraints.
        """
        sentences_text = "\n".join([f"{i+1}. {s}" for i, s in enumerate(sentences)])
        
        prompt = f"""Viết lại các câu sau thành MỘT đoạn văn liền mạch.
CHỈ dùng thông tin có sẵn, KHÔNG thêm bất kỳ thông tin nào khác.

Các câu cần nối:
{sentences_text}

Đoạn văn (chỉ dùng từ nối đơn giản):"""
        
        result = self._generate(prompt, max_new_tokens)
        
        result = result.strip()
        if result and result[-1] not in '.!?':
            result += '.'
        
        return result
    
    def summarize_strict(
        self,
        text: str,
        max_new_tokens: int = 150
    ) -> str:
        """
        Summarize with strict accuracy - no added information.
        """
        prompt = f"""Tóm tắt đoạn văn sau. GIỮ NGUYÊN số liệu và chủ ngữ.

Đoạn văn:
{text}

Tóm tắt (chỉ thông tin có trong văn bản gốc):"""
        
        result = self._generate(prompt, max_new_tokens)
        
        result = result.strip()
        if result and result[-1] not in '.!?':
            result += '.'
        
        return result
    
    def get_model_info(self) -> dict:
        """Return information about the model"""
        return {
            "model_name": self.MODEL_NAME,
            "mode": "STRICT PROMPTING",
            "description": "Qwen2.5-0.5B với Prompt nghiêm ngặt để giảm hallucination",
            "strict_rules": [
                "KHÔNG thêm thông tin mới",
                "GIỮ NGUYÊN chủ ngữ hành động",
                "GIỮ NGUYÊN số liệu chính xác",
                "KHÔNG thêm tính từ đánh giá",
                "KHÔNG suy diễn"
            ],
            "temperature": 0.2,  # Very low for accuracy
            "trade_off": "Ít mượt mà hơn, nhưng chính xác hơn",
            "model_size": "~500MB",
            "loaded": self._model is not None,
            "use_case": "Văn bản pháp lý, hành chính, cần độ chính xác cao"
        }


# Singleton instance for dependency injection
_qwen_strict_service: Optional[QwenStrictService] = None


def get_qwen_strict_service() -> QwenStrictService:
    """Get or create QwenStrictService singleton"""
    global _qwen_strict_service
    if _qwen_strict_service is None:
        _qwen_strict_service = QwenStrictService()
    return _qwen_strict_service
