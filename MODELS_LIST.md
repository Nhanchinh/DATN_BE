# 📋 DANH SÁCH ĐẦY ĐỦ CÁC MODELS EVALUATION

## ✅ TẤT CẢ MODELS HỖ TRỢ (9 models)

### 1️⃣ PhoBERT Extractive Models (An toàn 100% - Không hallucinate)

```json
{
  "model_name": "extractive_smart"
}
```
⭐ **RECOMMENDED** - Tự động trích xuất 30% câu quan trọng

```json
{
  "model_name": "extractive"
}
```
PhoBERT extractive cơ bản

```json
{
  "model_name": "extractive_chunked"
}
```
Trích xuất theo chunks (cân bằng coverage)

---

### 2️⃣ Generative Models (Abstractive)

```json
{
  "model_name": "vit5"
}
```
**Local ViT5 Finetuned** (~900MB) - Tiếng Việt tự nhiên

```json
{
  "model_name": "multilingual"
}
```
Giống `vit5` (alias)

```json
{
  "model_name": "bartpho"
}
```
**VinAI BARTpho** (~800MB) - Văn phong Việt chuẩn

---

### 3️⃣ Hybrid Models (Extractive + Generative) ⭐ TốT NHẤT!

```json
{
  "model_name": "hybrid"
}
```
**PhoBERT + mT5-XLSum** - Đa ngôn ngữ

```json
{
  "model_name": "hybrid_bartpho"
}
```
**PhoBERT + BARTpho** - VinAI ecosystem, văn phong Việt tự nhiên

```json
{
  "model_name": "hybrid_vit5"
}
```
⭐⭐⭐ **BEST CHO TIẾNG VIỆT!**
- **PhoBERT** (Extract câu quan trọng) → Không bịa
- **ViT5 Local** (Smooth từng câu) → Tự nhiên
- **Chia để trị**: Xử lý từng câu riêng lẻ → Giữ nguyên ý

---

## 🧪 JSON TEST CASES

### Test Hybrid BARTpho
```json
{
  "text": "Trí tuệ nhân tạo (AI) đang thay đổi cách chúng ta sống và làm việc. Công nghệ AI được ứng dụng rộng rãi trong nhiều lĩnh vực như y tế, giáo dục, giao thông, và tài chính. Trong y tế, AI giúp chẩn đoán bệnh chính xác hơn và phát triển thuốc mới nhanh chóng. Trong giáo dục, AI cá nhân hóa trải nghiệm học tập cho từng học sinh. Xe tự lái là một ví dụ điển hình của AI trong giao thông. Tuy nhiên, AI cũng đặt ra nhiều thách thức về đạo đức và an toàn dữ liệu mà chúng ta cần giải quyết.",
  "reference_summary": "AI đang được ứng dụng rộng rãi trong y tế, giáo dục, giao thông và tài chính. AI giúp cải thiện chẩn đoán bệnh, cá nhân hóa học tập và phát triển xe tự lái, nhưng cũng đặt ra thách thức về đạo đức.",
  "model_name": "hybrid_bartpho",
  "max_length": 150
}
```

### Test Hybrid ViT5 (BEST!)
```json
{
  "text": "Biến đổi khí hậu là một trong những thách thức lớn nhất của nhân loại. Nhiệt độ toàn cầu đang tăng do phát thải khí nhà kính từ hoạt động con người. Điều này gây ra băng tan ở cực, mực nước biển dâng, và thiên tai cực đoan như bão lũ, hạn hán ngày càng nghiêm trọng. Các quốc gia trên thế giới đã cam kết giảm phát thải và chuyển đổi sang năng lượng tái tạo. Tuy nhiên, tiến độ chuyển đổi vẫn còn chậm và cần sự nỗ lực lớn hơn từ tất cả các bên.",
  "reference_summary": "Biến đổi khí hậu do phát thải khí nhà kính gây băng tan, nước biển dâng và thiên tai. Các quốc gia cam kết giảm phát thải và dùng năng lượng tái tạo nhưng tiến độ còn chậm.",
  "model_name": "hybrid_vit5",
  "max_length": 150
}
```

### So Sánh Nhiều Models
```json
{
  "dataset_name": "vietnews",
  "model_names": ["extractive_smart", "hybrid_bartpho", "hybrid_vit5"],
  "sample_size": 10,
  "max_length": 150
}
```

---

## 📊 Model Comparison

| Model | Type | Size | Speed | Quality | Hallucinate Risk |
|-------|------|------|-------|---------|------------------|
| `extractive_smart` | Extractive | 400MB | ⚡⚡⚡ Fast | 😊 Good | ✅ ZERO |
| `extractive` | Extractive | 400MB | ⚡⚡⚡ Fast | 😐 OK | ✅ ZERO |
| `extractive_chunked` | Extractive | 400MB | ⚡⚡⚡ Fast | 😊 Good | ✅ ZERO |
| `vit5` | Generative | 900MB | ⚡⚡ Medium | 😊 Good | ⚠️ LOW |
| `bartpho` | Generative | 800MB | ⚡⚡ Medium | 😊 Good | ⚠️ LOW |
| `hybrid` | Hybrid | 1.6GB | ⚡ Slow | 🎉 Great | ⚠️ VERY LOW |
| `hybrid_bartpho` | Hybrid | 1.2GB | ⚡ Slow | 🎉 Great | ✅ VERY LOW |
| **`hybrid_vit5`** | **Hybrid** | **1.3GB** | **⚡ Slow** | **🔥 BEST** | **✅ VERY LOW** |

**Notes:**
- **Extractive**: Chỉ trích câu gốc → An toàn tuyệt đối
- **Generative**: Viết lại → Tự nhiên nhưng có thể hallucinate
- **Hybrid**: Kết hợp 2 loại → Cân bằng an toàn & tự nhiên

---

## 🚀 Performance Tips

### Lần Đầu Load (Chậm)
- `extractive_smart`: ~10-20s (load PhoBERT)
- `vit5`: ~15-30s (load ViT5 local)
- `bartpho`: ~15-30s (load BARTpho)
- `hybrid_vit5`: ~30-60s (load cả 2 models)
- **BERTScore**: ~90-100s (load BERT multilingual lần đầu)

### Lần Sau (Nhanh)
- **Single evaluation**: 2-5s
- **Batch 10 samples**: ~30-60s
- **Batch 100 samples**: ~5-10 phút

### RAM Usage
- Extractive only: ~2GB
- Hybrid: ~4-6GB
- With BERTScore: ~6-8GB

---

## ✅ RECOMMENDED WORKFLOW

### 1. Development/Testing
```
extractive_smart → Nhanh, an toàn
```

### 2. Production (Tiếng Việt)
```
hybrid_vit5 → Chất lượng tốt nhất
```

### 3. Batch Evaluation (So sánh)
```json
{
  "model_names": ["extractive_smart", "hybrid_bartpho", "hybrid_vit5"]
}
```

### 4. VietNews Benchmark
```json
{
  "dataset_name": "vietnews",
  "model_names": ["hybrid_vit5"],
  "sample_size": 100
}
```
