"""
Vietnamese Summarization Service - Using Finetuned ViT5
Model được fine-tune với prefix "làm mượt: " để viết lại văn bản mượt mà.
"""

import logging
import random
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
        Cơ chế "Bảo hiểm Số liệu" nâng cao.
        
        Kiểm tra 2 loại lỗi:
        1. Mất số: Câu gốc có "12.000" mà ViT5 bỏ mất
        2. Sai số thứ tự: "Giai đoạn 1" -> "Giai đoạn 2" (NGUY HIỂM!)
        
        -> Nếu phát hiện lỗi: Vứt câu ViT5, dùng lại câu gốc.
        
        Args:
            original: Câu gốc từ PhoBERT
            generated: Câu ViT5 vừa sinh ra
            
        Returns:
            str: Câu an toàn (không bị sai lệch số liệu)
        """
        # ========== CHECK 1: Mất số ==========
        numbers_original = set(re.findall(r'\d+', original))
        numbers_generated = set(re.findall(r'\d+', generated))
        
        if numbers_original and not numbers_original.issubset(numbers_generated):
            missing_numbers = numbers_original - numbers_generated
            logger.warning(
                f"⚠️ ViT5 làm mất số liệu: {missing_numbers}. Fallback về câu gốc."
            )
            return original
        
        # ========== CHECK 2: Sai số thứ tự (Ordinal Hallucination) ==========
        # Tìm pattern "Giai đoạn X", "Bước X", "Phần X", "Cấp X", "Lớp X"
        ordinal_patterns = [
            r'giai\s*đoạn\s*(\d+)',
            r'bước\s*(\d+)',
            r'phần\s*(\d+)',
            r'cấp\s*(\d+)',
            r'lớp\s*(\d+)',
            r'năm\s*(\d{4})',  # Năm cũng quan trọng
        ]
        
        for pattern in ordinal_patterns:
            orig_matches = re.findall(pattern, original.lower())
            gen_matches = re.findall(pattern, generated.lower())
            
            # Nếu câu gốc có ordinal mà câu sinh ra có ordinal KHÁC -> Lỗi!
            if orig_matches and gen_matches:
                # So sánh xem có bị đổi số không
                for orig_num in orig_matches:
                    if orig_num not in gen_matches:
                        # Số thứ tự bị thay đổi -> Nguy hiểm!
                        logger.warning(
                            f"🚨 ViT5 sai số thứ tự: '{pattern}' {orig_num} -> {gen_matches}. "
                            f"Fallback về câu gốc."
                        )
                        return original
        
        # ========== CHECK 3: Số mới xuất hiện mà không có trong gốc ==========
        # Nếu ViT5 tự bịa ra số mới -> Cũng là hallucination
        new_numbers = numbers_generated - numbers_original
        # Chỉ cảnh báo nếu số mới là số quan trọng (> 2 chữ số hoặc là năm)
        suspicious_new = [n for n in new_numbers if len(n) >= 2 or int(n) > 100]
        if suspicious_new and numbers_original:
            logger.warning(
                f"⚠️ ViT5 tự bịa số mới: {suspicious_new}. Fallback về câu gốc."
            )
            return original
        
        return generated  # Đủ an toàn, dùng câu ViT5
    
    def smooth_sentences(self, sentences: list) -> str:
        """
        Làm mượt TỪNG CÂU trong list, rồi ghép lại thành đoạn văn liền mạch.
        
        Pipeline "Chia để trị" nâng cao:
        1. Xử lý từng câu riêng lẻ qua ViT5
        2. Thay thế từ lặp bằng từ đồng nghĩa (Dynamic Synonym)
        3. Thêm từ nối dựa trên sentiment (Sentiment-based Linking)
        4. Hậu xử lý để fix format + polish văn bản
        
        Args:
            sentences: List các câu đã được PhoBERT trích xuất
            
        Returns:
            str: Đoạn văn mượt mà, liền mạch
        """
        self._load_model()
        
        smoothed_parts = []
        
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 10:  # Bỏ qua câu rác quá ngắn
                smooth = self.smooth_sentence(sent)
                if smooth:
                    smoothed_parts.append(smooth)
        
        # ========== BƯỚC 2: DYNAMIC SYNONYM REPLACEMENT ==========
        # Thay thế từ lặp bằng từ đồng nghĩa để tránh monotonous
        smoothed_parts = self._dynamic_synonym_replace(smoothed_parts)
        
        # ========== BƯỚC 3: SENTIMENT-BASED LINKING ==========
        # Thêm từ nối phù hợp giữa các câu dựa trên sentiment
        smoothed_parts = self._add_dynamic_connectors(smoothed_parts)
        
        # Ghép lại thành đoạn văn
        final_text = " ".join(smoothed_parts)
        
        # Hậu xử lý: Xóa các từ nối vô nghĩa ở đầu câu đầu tiên
        unwanted_starts = [
            "Theo đó,", "Theo đó ", 
            "Bên cạnh đó,", "Bên cạnh đó ",
            "Thêm vào đó,", "Thêm vào đó ",
            "Ngoài ra,", "Ngoài ra ",
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
    
    # ==================== KHO TỪ ĐIỂN ĐỒNG NGHĨA ĐỘNG ====================
    
    # Kho từ điển đồng nghĩa (dễ mở rộng)
    SYNONYM_DICT = {
        "cho biết": ["nhận định", "chia sẻ", "đánh giá", "khẳng định", "nhấn mạnh", "nêu rõ"],
        "nói rằng": ["cho hay", "phát biểu", "bày tỏ", "tuyên bố"],
        "dự kiến": ["theo kế hoạch", "được kỳ vọng", "ước tính"],
        "triển khai": ["thực hiện", "áp dụng", "tiến hành"],
        "đánh giá": ["nhận xét", "đánh giá", "ghi nhận"],
    }
    
    def _dynamic_synonym_replace(self, sentences: List[str]) -> List[str]:
        """
        Thay thế từ lặp bằng từ đồng nghĩa (Dynamic Synonym Replacement).
        
        Logic:
        - Theo dõi các từ đã dùng trong các câu trước
        - Nếu câu hiện tại có từ đã dùng -> Thay thế bằng từ đồng nghĩa random
        - Reset bộ nhớ sau mỗi 3 câu để tự nhiên hơn
        
        Args:
            sentences: List các câu đã làm mượt
            
        Returns:
            List[str]: Các câu đã được thay thế từ lặp
        """
        refined_sentences = []
        used_words = set()
        
        for i, sent in enumerate(sentences):
            # Reset bộ nhớ sau mỗi 3 câu để văn phong tự nhiên
            if i % 3 == 0:
                used_words.clear()
            
            new_sent = sent
            
            for key, replacements in self.SYNONYM_DICT.items():
                # Kiểm tra từ khóa trong câu (không phân biệt hoa thường)
                if key.lower() in new_sent.lower():
                    # Nếu từ này vừa dùng ở câu trước -> THAY THẾ
                    if key in used_words:
                        # Chọn random 1 từ thay thế chưa dùng
                        available = [w for w in replacements if w not in used_words]
                        if available:
                            replacement = random.choice(available)
                            # Thay thế (giữ nguyên viết hoa nếu ở đầu câu)
                            pattern = re.compile(re.escape(key), re.IGNORECASE)
                            new_sent = pattern.sub(replacement, new_sent, count=1)
                            used_words.add(replacement)
                            logger.debug(f"🔄 Synonym: '{key}' -> '{replacement}'")
                    else:
                        used_words.add(key)
            
            refined_sentences.append(new_sent)
        
        return refined_sentences
    
    # ==================== SENTIMENT-BASED LINKING ====================
    
    # Từ khóa để phát hiện sentiment
    POSITIVE_KEYWORDS = [
        "tốt", "quan trọng", "ủng hộ", "đánh giá cao", "lợi ích", 
        "thành công", "hiệu quả", "tiến bộ", "phát triển", "thuận lợi",
        "ưu điểm", "cải thiện", "nâng cao", "khuyến khích"
    ]
    
    NEGATIVE_KEYWORDS = [
        "lo ngại", "khó khăn", "thiếu", "hạn chế", "rào cản",
        "tuy nhiên", "nhược điểm", "thách thức", "vấn đề", "không đủ",
        "chưa", "mặc dù", "trở ngại", "bất cập"
    ]
    
    # Từ nối theo sentiment transition
    CONNECTORS = {
        "positive_to_negative": ["Tuy nhiên,", "Mặc dù vậy,", "Song,", "Dẫu vậy,"],
        "negative_to_positive": ["Mặt khác,", "Trái lại,", "Ngược lại,"],
        "same_positive": ["Đồng thời,", "Bên cạnh đó,", "Ngoài ra,", "Hơn nữa,"],
        "same_negative": ["Thêm vào đó,", "Đáng lo ngại hơn,", "Cũng có ý kiến rằng,"],
        "neutral": ["Theo đó,", "Cụ thể,", "Về mặt này,"]
    }
    
    def _get_sentiment(self, text: str) -> int:
        """
        Phân tích sentiment đơn giản dựa trên từ khóa.
        
        Returns:
            1: Tích cực (Positive)
            -1: Tiêu cực (Negative)
            0: Trung tính (Neutral)
        """
        text_lower = text.lower()
        
        pos_count = sum(1 for kw in self.POSITIVE_KEYWORDS if kw in text_lower)
        neg_count = sum(1 for kw in self.NEGATIVE_KEYWORDS if kw in text_lower)
        
        if pos_count > neg_count:
            return 1
        elif neg_count > pos_count:
            return -1
        return 0
    
    def _add_dynamic_connectors(self, sentences: List[str]) -> List[str]:
        """
        Thêm từ nối phù hợp giữa các câu dựa trên sentiment.
        
        Logic nâng cao:
        - Phân tích sentiment của câu trước và câu hiện tại
        - Nếu chuyển từ positive -> negative: Thêm "Tuy nhiên,"
        - Nếu câu bắt đầu bằng "Bên cạnh" nhưng sentiment đổi chiều: Thay bằng "Tuy nhiên,"
        - Nếu cùng chiều: Thêm "Đồng thời," hoặc "Ngoài ra,"
        
        Args:
            sentences: List các câu
            
        Returns:
            List[str]: Các câu đã được thêm/sửa từ nối
        """
        if len(sentences) <= 1:
            return sentences
        
        # Từ nối YẾU cần được thay thế khi sentiment thay đổi
        WEAK_CONNECTORS = [
            "bên cạnh đó", "bên cạnh", "ngoài ra", "thêm vào đó", 
            "đồng thời", "hơn nữa"
        ]
        
        result = [sentences[0]]  # Câu đầu tiên giữ nguyên
        
        for i in range(1, len(sentences)):
            prev_sent = sentences[i - 1]
            curr_sent = sentences[i]
            
            # Lấy sentiment
            prev_sentiment = self._get_sentiment(prev_sent)
            curr_sentiment = self._get_sentiment(curr_sent)
            
            # ========== XỬ LÝ ĐẶC BIỆT: "Bên cạnh" khi sentiment đổi chiều ==========
            # Nếu câu bắt đầu bằng từ nối yếu nhưng sentiment đổi chiều -> Thay thế
            curr_lower = curr_sent.lower()
            starts_with_weak = any(curr_lower.startswith(wc) for wc in WEAK_CONNECTORS)
            
            if starts_with_weak and prev_sentiment > 0 and curr_sentiment < 0:
                # Tìm và thay thế từ nối yếu bằng từ nối mạnh
                for wc in WEAK_CONNECTORS:
                    if curr_lower.startswith(wc):
                        # Xóa từ nối yếu
                        rest_of_sentence = curr_sent[len(wc):].lstrip(", ")
                        # Thêm từ nối mạnh
                        strong_connector = random.choice(self.CONNECTORS["positive_to_negative"])
                        curr_sent = f"{strong_connector} {rest_of_sentence[0].lower() + rest_of_sentence[1:] if rest_of_sentence else ''}"
                        logger.debug(f"🔄 Replaced weak connector '{wc}' with '{strong_connector}'")
                        break
                result.append(curr_sent)
                continue
            
            # ========== LOGIC CONNECTOR THÔNG THƯỜNG ==========
            # Kiểm tra xem câu hiện tại đã có từ nối chưa
            has_connector = any(
                curr_sent.lower().startswith(conn.lower().rstrip(','))
                for connectors in self.CONNECTORS.values()
                for conn in connectors
            )
            
            # Chỉ thêm từ nối nếu câu chưa có
            if not has_connector:
                connector = ""
                
                # Quyết định loại từ nối dựa trên sentiment transition
                if prev_sentiment > 0 and curr_sentiment < 0:
                    connector = random.choice(self.CONNECTORS["positive_to_negative"])
                elif prev_sentiment < 0 and curr_sentiment > 0:
                    connector = random.choice(self.CONNECTORS["negative_to_positive"])
                elif prev_sentiment == curr_sentiment and prev_sentiment > 0:
                    # Chỉ thêm từ nối cho một số câu (không phải tất cả) để tự nhiên
                    if random.random() < 0.6:  # 60% chance
                        connector = random.choice(self.CONNECTORS["same_positive"])
                elif prev_sentiment == curr_sentiment and prev_sentiment < 0:
                    if random.random() < 0.5:  # 50% chance
                        connector = random.choice(self.CONNECTORS["same_negative"])
                
                if connector:
                    # Thêm từ nối và viết hoa chữ cái đầu tiên của câu gốc
                    curr_sent_adjusted = curr_sent[0].lower() + curr_sent[1:] if curr_sent else curr_sent
                    curr_sent = f"{connector} {curr_sent_adjusted}"
                    logger.debug(f"🔗 Added connector: '{connector}'")
            
            result.append(curr_sent)
        
        return result
    
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
