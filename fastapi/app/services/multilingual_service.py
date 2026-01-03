"""
Vietnamese Summarization Service - Using Finetuned ViT5
Model được fine-tune với prefix "làm mượt: " để viết lại văn bản mượt mà.
"""

import logging
import re
from typing import Optional, Tuple, List

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from app.utils.text_processor import TextProcessor, get_text_processor

logger = logging.getLogger(__name__)


class MultilingualSummarizationService:
    """
    Vietnamese summarization service using local finetuned ViT5.
    
    Model được fine-tune với prefix "làm mượt: " để viết lại văn bản.
    
    QUAN TRỌNG: Prefix "làm mượt: " phải được thêm vào đầu input,
    nếu không model sẽ không hoạt động đúng.
    """
    
    MODEL_NAME = "AI_Models/my_vit5_model"
    PREFIX = "làm mượt: "  # Prefix bắt buộc - model được train với prefix này
    
    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._text_processor: TextProcessor = get_text_processor()
        
        logger.info(f"ViT5 SummarizationService initialized. Device: {self._device}")
    
    def _load_model(self) -> None:
        """Lazy load ViT5 model"""
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
        max_length: int = 150,
        min_length: int = 30,
        language: str = "vi"
    ) -> Tuple[str, str]:
        """
        Summarize Vietnamese text.
        
        Args:
            text: Input Vietnamese text to summarize
            max_length: Maximum summary length (in tokens)
            min_length: Minimum summary length (in tokens)
            language: Language hint (vi for Vietnamese)
            
        Returns:
            Tuple[str, str]: (raw_summary, processed_summary)
        """
        self._load_model()
        
        # PRE-PROCESSING
        cleaned_text = self._text_processor.preprocess(text)
        
        # QUAN TRỌNG: Thêm prefix "làm mượt: " - model được train với prefix này
        # Thiếu prefix thì model sẽ trả kết quả linh tinh
        input_text = self.PREFIX + cleaned_text
        
        inputs = self._tokenizer(
            input_text,
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
        
        # POST-PROCESSING (lighter for Vietnamese)
        processed_summary = raw_summary.strip()
        
        # Ensure proper ending
        if processed_summary and processed_summary[-1] not in '.!?':
            processed_summary += '.'
        
        return raw_summary, processed_summary
    
    def smooth_sentence(self, sentence: str) -> str:
        """
        Làm mượt MỘT câu đơn lẻ.
        
        Đây là cách đúng để dùng model vì model được train với input là 1 câu.
        Có cơ chế "Bảo hiểm Số liệu" - nếu ViT5 làm mất số thì fallback về câu gốc.
        
        Args:
            sentence: Một câu tiếng Việt cần làm mượt
            
        Returns:
            str: Câu đã được làm mượt (hoặc câu gốc nếu ViT5 làm mất thông tin)
        """
        self._load_model()
        
        original_sentence = sentence.strip()
        
        # Bỏ qua câu quá ngắn
        if len(original_sentence) < 10:
            return original_sentence
        
        # Thêm prefix bắt buộc
        input_text = self.PREFIX + original_sentence
        
        inputs = self._tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self._device)
        
        # Tính min_length động: ít nhất 70% độ dài input (token)
        # Điều này NGĂN model cắt xén quá đà
        input_token_count = inputs["input_ids"].shape[1]
        dynamic_min_length = max(20, int(input_token_count * 0.7))
        
        with torch.no_grad():
            outputs = self._model.generate(
                inputs["input_ids"],
                max_length=512,  # Cho phép viết dài
                min_length=dynamic_min_length,  # Ép viết đủ dài
                num_beams=8,  # Tìm kiếm kỹ hơn
                length_penalty=2.0,  # Khuyến khích viết dài
                repetition_penalty=1.2,  # Tránh lặp từ
                no_repeat_ngram_size=2,  # Giảm xuống để không cắt nhầm
                early_stopping=True
            )
        
        result = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        result = result.strip()
        
        # Xóa các ký tự thừa ở đầu (*, -, số thứ tự...)
        result = result.lstrip('*-•–—. ')
        # Nhưng KHÔNG lstrip số vì số có thể là thông tin quan trọng!
        
        # CƠ CHẾ "BẢO HIỂM SỐ LIỆU"
        # Nếu câu gốc có số mà ViT5 làm mất -> Fallback về câu gốc
        result = self._safety_check_numbers(original_sentence, result)
        
        # Đảm bảo kết thúc đúng
        if result and result[-1] not in '.!?':
            result += '.'
            
        return result
    
    def _safety_check_numbers(self, original: str, generated: str) -> str:
        """
        Cơ chế "Bảo hiểm Số liệu".
        
        Nếu câu gốc có số mà câu ViT5 sinh ra bị mất số đó
        -> Vứt câu ViT5 đi, dùng lại câu gốc cho an toàn.
        
        Args:
            original: Câu gốc từ PhoBERT
            generated: Câu ViT5 vừa sinh ra
            
        Returns:
            str: Câu an toàn (có đủ số liệu)
        """
        # Tìm tất cả con số trong câu gốc
        numbers_original = set(re.findall(r'\d+', original))
        
        # Tìm tất cả con số trong câu sinh ra
        numbers_generated = set(re.findall(r'\d+', generated))
        
        # Nếu câu gốc có số mà câu sinh ra không có -> Có biến!
        if numbers_original and not numbers_original.issubset(numbers_generated):
            missing_numbers = numbers_original - numbers_generated
            logger.warning(
                f"⚠️ ViT5 làm mất số liệu: {missing_numbers}. Fallback về câu gốc."
            )
            return original  # Dùng lại câu gốc cho an toàn
        
        return generated  # Nếu đủ số thì dùng câu ViT5
    
    def smooth_sentences(self, sentences: list) -> str:
        """
        Làm mượt TỪNG CÂU trong list, rồi ghép lại thành đoạn văn.
        
        Đây là phương pháp "Chia để trị" - đảm bảo không mất ý:
        1. Xử lý từng câu riêng lẻ qua ViT5
        2. Ghép các câu đã làm mượt lại
        3. Hậu xử lý để loại bỏ từ nối thừa + fix format
        
        Args:
            sentences: List các câu đã được PhoBERT trích xuất
            
        Returns:
            str: Đoạn văn đã làm mượt
        """
        self._load_model()
        
        smoothed_parts = []
        
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 10:  # Bỏ qua câu rác quá ngắn
                smooth = self.smooth_sentence(sent)
                if smooth:
                    smoothed_parts.append(smooth)
        
        # Ghép lại thành đoạn văn
        final_text = " ".join(smoothed_parts)
        
        # Hậu xử lý: Xóa các từ nối vô nghĩa ở đầu câu đầu tiên
        unwanted_starts = [
            "Theo đó,", "Theo đó ", 
            "Bên cạnh đó,", "Bên cạnh đó ",
            "Thêm vào đó,", "Thêm vào đó ",
            "Ngoài ra,", "Ngoài ra ",
            "Tuy nhiên,", "Tuy nhiên ",
            "Do đó,", "Do đó ",
        ]
        for phrase in unwanted_starts:
            if final_text.startswith(phrase):
                final_text = final_text[len(phrase):].strip()
                break
        
        # Viết hoa chữ cái đầu
        if final_text:
            final_text = final_text[0].upper() + final_text[1:]
        
        # ÁP DỤNG POST-PROCESSING POLISH (Fix format -> 10/10)
        final_text = self._post_process_polish(final_text)
        
        return final_text
    
    def _post_process_polish(self, text: str) -> str:
        """
        Hậu xử lý để đạt điểm 10/10.
        
        Fix 2 lỗi format phổ biến:
        1. Viết hoa sau dấu chấm (". b" -> ". B")
        2. Số liệu bị tách (4. 0 -> 4.0, 25. 000 -> 25.000)
        
        Args:
            text: Văn bản cần polish
            
        Returns:
            str: Văn bản đã được polish hoàn hảo
        """
        if not text:
            return text
        
        # 1. Fix lỗi số liệu bị tách (PHẢI LÀM TRƯỚC)
        # 4. 0 -> 4.0, 25. 000 -> 25.000
        text = re.sub(r'(\d+)\.\s+(\d+)', r'\1.\2', text)
        
        # 2. Fix lỗi viết hoa sau dấu chấm
        # ". b" -> ". B"
        def capitalize_after_period(match):
            return match.group(1) + match.group(2).upper()
        
        text = re.sub(r'(\.\s+)([a-zàáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ])', 
                      capitalize_after_period, text)
        
        return text
    
    def get_model_info(self) -> dict:
        """Return information about the model"""
        return {
            "model_name": self.MODEL_NAME,
            "prefix": self.PREFIX,
            "description": "ViT5 fine-tuned với prefix 'làm mượt:' để viết lại văn bản mượt mà",
            "supported_languages": ["vi"],
            "model_size": "~900MB",
            "loaded": self._model is not None,
            "type": "local finetuned",
            "note": "Nên dùng smooth_sentences() để xử lý từng câu riêng lẻ"
        }


# Singleton instance for dependency injection
_multilingual_service: Optional[MultilingualSummarizationService] = None


def get_multilingual_service() -> MultilingualSummarizationService:
    """Get or create MultilingualSummarizationService singleton"""
    global _multilingual_service
    if _multilingual_service is None:
        _multilingual_service = MultilingualSummarizationService()
    return _multilingual_service
