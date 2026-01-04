# API Testing Guide - Evaluation Endpoints

## Prerequisites

```bash
# 1. Start server
cd d:\DATN_HT_DGTTVB\DATN_BE\fastapi
uvicorn app.main:app --reload

# 2. Get authentication token
# Login to get token first
```

---

## 🔐 Step 1: Login & Get Token

### Register User (if needed)
```bash
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d "{\"email\": \"test@example.com\", \"password\": \"secret123\", \"full_name\": \"Test User\"}"
```

### Login
```bash
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=test@example.com&password=secret123"
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

**⚠️ IMPORTANT:** Lưu `access_token` để dùng cho các requests sau!

---

## 📊 Evaluation API Endpoints

### 1️⃣ POST /evaluate/single - Đánh giá 1 văn bản

**Request:**
```bash
curl -X POST http://localhost:8000/evaluate/single \
  -H "Authorization: Bearer YOUR_TOKEN_HERE" \
  -H "Content-Type: application/json" \
  -d @- << 'EOF'
{
  "text": "Việt Nam là một quốc gia nằm ở phía đông bán đảo Đông Dương thuộc khu vực Đông Nam Á. Việt Nam có diện tích 331.212 km², dân số gần 100 triệu người. Thủ đô của Việt Nam là Hà Nội, thành phố lớn nhất là Thành phố Hồ Chí Minh. Việt Nam có bờ biển dài hơn 3.260 km và giáp với Trung Quốc, Lào và Campuchia.",
  "reference_summary": "Việt Nam là quốc gia Đông Nam Á, diện tích 331.212 km², dân số gần 100 triệu. Thủ đô Hà Nội, TP lớn nhất là TP Hồ Chí Minh.",
  "model_name": "extractive_smart",
  "max_length": 150
}
EOF
```

**JSON Body:**
```json
{
  "text": "Việt Nam là một quốc gia nằm ở phía đông bán đảo Đông Dương thuộc khu vực Đông Nam Á. Việt Nam có diện tích 331.212 km², dân số gần 100 triệu người. Thủ đô của Việt Nam là Hà Nội, thành phố lớn nhất là Thành phố Hồ Chí Minh. Việt Nam có bờ biển dài hơn 3.260 km và giáp với Trung Quốc, Lào và Campuchia.",
  "reference_summary": "Việt Nam là quốc gia Đông Nam Á, diện tích 331.212 km², dân số gần 100 triệu. Thủ đô Hà Nội, TP lớn nhất là TP Hồ Chí Minh.",
  "model_name": "extractive_smart",
  "max_length": 150
}
```

**Model names hỗ trợ:**
- `extractive_smart` ⭐ (Recommended - PhoBERT)
- `extractive` (PhoBERT cơ bản)
- `extractive_chunked` (PhoBERT chunks)
- `vit5` (ViT5 model)
- `multilingual` (ViT5)
- `bartpho` (BARTpho VinAI)
- `hybrid` (PhoBERT + mT5)

**Expected Response:**
```json
{
  "generated_summary": "Việt Nam là một quốc gia nằm ở phía đông bán đảo Đông Dương thuộc khu vực Đông Nam Á. Thủ đô của Việt Nam là Hà Nội, thành phố lớn nhất là Thành phố Hồ Chí Minh.",
  "metrics": {
    "rouge1": 0.65,
    "rouge2": 0.45,
    "rougeL": 0.58,
    "bleu": 0.42,
    "bert_score": 0.78,
    "processing_time_ms": 1250
  },
  "evaluation_id": "507f1f77bcf86cd799439011"
}
```

---

### 2️⃣ POST /evaluate/batch - Đánh giá hàng loạt

**⚠️ WARNING:** Batch evaluation chạy background task, có thể mất 5-10 phút cho 100 samples!

**Request:**
```bash
curl -X POST http://localhost:8000/evaluate/batch \
  -H "Authorization: Bearer YOUR_TOKEN_HERE" \
  -H "Content-Type: application/json" \
  -d @- << 'EOF'
{
  "dataset_name": "vietnews",
  "model_names": ["extractive_smart"],
  "sample_size": 10,
  "max_length": 150
}
EOF
```

**JSON Body:**
```json
{
  "dataset_name": "vietnews",
  "model_names": ["extractive_smart"],
  "sample_size": 10,
  "max_length": 150
}
```

**Tips:**
- Start với `sample_size: 10` để test nhanh
- Tối đa 3 models cùng lúc: `["extractive_smart", "vit5", "bartpho"]`
- sample_size: 10-1000 (recommend 100 cho production)

**Expected Response:**
```json
{
  "evaluation_id": "507f1f77bcf86cd799439012",
  "status": "queued",
  "message": "Batch evaluation queued for 1 model(s). Use /evaluate/progress/507f1f77bcf86cd799439012 to track progress."
}
```

---

### 3️⃣ GET /evaluate/progress/{evaluation_id} - Track progress

**Request:**
```bash
curl -X GET http://localhost:8000/evaluate/progress/507f1f77bcf86cd799439012 \
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

**Expected Response (Running):**
```json
{
  "evaluation_id": "507f1f77bcf86cd799439012",
  "status": "running",
  "progress": 45,
  "current_sample": 5,
  "total_samples": 10,
  "overall_metrics": null,
  "error_message": null
}
```

**Expected Response (Completed):**
```json
{
  "evaluation_id": "507f1f77bcf86cd799439012",
  "status": "completed",
  "progress": 100,
  "current_sample": 10,
  "total_samples": 10,
  "overall_metrics": {
    "avg_rouge1": 0.42,
    "avg_rouge2": 0.28,
    "avg_rougeL": 0.35,
    "avg_bleu": 0.22,
    "avg_bert_score": 0.70,
    "avg_processing_time_ms": 135,
    "total_samples": 10
  },
  "error_message": null
}
```

**💡 Frontend Usage:**
```javascript
// Poll every 5 seconds
setInterval(async () => {
  const response = await fetch(`/evaluate/progress/${evaluationId}`);
  const data = await response.json();
  
  if (data.status === 'completed') {
    // Show results
    console.log(data.overall_metrics);
  } else if (data.status === 'failed') {
    // Show error
    console.error(data.error_message);
  } else {
    // Update progress bar
    updateProgress(data.progress);
  }
}, 5000);
```

---

### 4️⃣ POST /evaluate/compare - So sánh models

**Request:**
```bash
curl -X POST http://localhost:8000/evaluate/compare \
  -H "Authorization: Bearer YOUR_TOKEN_HERE" \
  -H "Content-Type: application/json" \
  -d @- << 'EOF'
{
  "model_names": ["extractive_smart", "vit5", "bartpho"],
  "dataset_name": "vietnews",
  "limit": 10
}
EOF
```

**JSON Body:**
```json
{
  "model_names": ["extractive_smart", "vit5", "bartpho"],
  "dataset_name": "vietnews",
  "limit": 10
}
```

**Expected Response:**
```json
{
  "comparisons": [
    {
      "model_name": "extractive_smart",
      "avg_rouge1": 0.45,
      "avg_rouge2": 0.30,
      "avg_rougeL": 0.38,
      "avg_bleu": 0.25,
      "avg_bert_score": 0.72,
      "evaluation_count": 5
    },
    {
      "model_name": "vit5",
      "avg_rouge1": 0.48,
      "avg_rouge2": 0.32,
      "avg_rougeL": 0.40,
      "avg_bleu": 0.28,
      "avg_bert_score": 0.75,
      "evaluation_count": 3
    },
    {
      "model_name": "bartpho",
      "avg_rouge1": 0.50,
      "avg_rouge2": 0.35,
      "avg_rougeL": 0.42,
      "avg_bleu": 0.30,
      "avg_bert_score": 0.78,
      "evaluation_count": 2
    }
  ],
  "dataset_name": "vietnews"
}
```

---

### 5️⃣ GET /evaluate/history - Lịch sử đánh giá

**Request:**
```bash
curl -X GET "http://localhost:8000/evaluate/history?limit=5&model_name=extractive_smart" \
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

**Query Parameters:**
- `limit` (optional): Số lượng records (default 10)
- `model_name` (optional): Filter theo model
- `dataset_name` (optional): Filter theo dataset

**Expected Response:**
```json
{
  "evaluations": [
    {
      "evaluation_id": "507f1f77bcf86cd799439011",
      "model_name": "extractive_smart",
      "dataset_name": "manual",
      "created_at": "2026-01-04T12:30:00",
      "overall_metrics": {
        "avg_rouge1": 0.65,
        "avg_rouge2": 0.45,
        "avg_rougeL": 0.58,
        "avg_bleu": 0.42,
        "avg_bert_score": 0.78,
        "avg_processing_time_ms": 1250,
        "total_samples": 1
      },
      "status": "completed"
    },
    {
      "evaluation_id": "507f1f77bcf86cd799439012",
      "model_name": "extractive_smart",
      "dataset_name": "vietnews",
      "created_at": "2026-01-04T12:25:00",
      "overall_metrics": {
        "avg_rouge1": 0.42,
        "avg_rouge2": 0.28,
        "avg_rougeL": 0.35,
        "avg_bleu": 0.22,
        "avg_bert_score": 0.70,
        "avg_processing_time_ms": 135,
        "total_samples": 10
      },
      "status": "completed"
    }
  ],
  "total_count": 2
}
```

---

### 6️⃣ GET /evaluate/datasets - List datasets

**Request:**
```bash
curl -X GET http://localhost:8000/evaluate/datasets \
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

**Expected Response:**
```json
{
  "datasets": [
    {
      "name": "VietNews",
      "source": "vietnews",
      "total_samples": 4246,
      "description": "Vietnamese news summarization dataset from HuggingFace"
    }
  ]
}
```

---

## 🔥 Quick Test Workflow

### Step-by-step testing:

```bash
# 1. Login
TOKEN=$(curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=test@example.com&password=secret123" \
  | jq -r '.access_token')

echo "Token: $TOKEN"

# 2. Test single evaluation
curl -X POST http://localhost:8000/evaluate/single \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"text": "Đây là văn bản test ngắn để kiểm tra hệ thống đánh giá tóm tắt văn bản tiếng Việt với các metrics như ROUGE, BLEU và BERTScore.", "reference_summary": "Test hệ thống đánh giá tóm tắt.", "model_name": "extractive_smart"}' \
  | jq

# 3. Test batch (small sample)
curl -X POST http://localhost:8000/evaluate/batch \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"dataset_name": "vietnews", "model_names": ["extractive_smart"], "sample_size": 5}' \
  | jq

# 4. Get history
curl -X GET "http://localhost:8000/evaluate/history?limit=5" \
  -H "Authorization: Bearer $TOKEN" \
  | jq
```

---

## 📝 Notes

### Performance Expectations:
- **Single evaluation**: 1-3 seconds (với BERTScore)
- **Batch 10 samples**: ~1-2 minutes
- **Batch 100 samples**: ~5-10 minutes
- **Batch 1000 samples**: ~50-90 minutes

### Metrics Interpretation:
- **ROUGE-1**: Unigram overlap (0-1, higher is better)
- **ROUGE-2**: Bigram overlap (0-1, higher is better)
- **ROUGE-L**: Longest common subsequence (0-1, higher is better)
- **BLEU**: Precision-based metric (0-1, higher is better)
- **BERTScore**: Semantic similarity (0-1, higher is better)

### Troubleshooting:
1. **401 Unauthorized**: Token expired, login again
2. **400 Unknown model**: Check model_name spelling
3. **500 Server error**: Check server logs for details
4. **Batch timeout**: Normal! Use GET /evaluate/progress to track

---

## 🎯 Next Steps

After backend testing works:
1. ✅ Verify all endpoints work
2. 🎨 Build frontend UI
3. 📊 Create charts in Analytics page
4. 🚀 Deploy to production
