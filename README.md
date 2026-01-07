# Viterbox-vLLM 🚀

**High-performance vLLM backend for Viterbox TTS**

Viterbox-vLLM là phiên bản **tối ưu hiệu năng** của Chatterbox/Viterbox TTS, sử dụng **vLLM** làm backend suy luận. Phiên bản này được thiết kế để **tăng tốc độ suy luận lên ~4× so với bản thông thường**, đồng thời hỗ trợ cả **inference đơn mẫu** và **batch inference**.

---

## ✨ Tính năng chính

| Tính năng | Mô tả |
|-----------|-------|
| ⚡ **Hiệu năng cao** | Nhanh hơn ~4× so với backend truyền thống (PyTorch eager) |
| 🔥 **vLLM Backend** | Inference hiệu quả và ổn định |
| 📦 **Đa dạng inference** | Hỗ trợ inference 1 mẫu và batch |
| 🧠 **Tương thích** | Hoạt động với model Viterbox |
| 🧪 **Dễ sử dụng** | Có sẵn notebook hướng dẫn (`test.ipynb`) |
| 🛠 **Tích hợp dễ dàng** | Dễ dàng tích hợp vào pipeline TTS hiện có |

---

## 📂 Cấu trúc Repository

```
.
├── src/
│   └── chatterbox_vllm/
│       └── ...
├── tts.py                 # Core TTS implementation
├── test.ipynb             # Notebook hướng dẫn sử dụng
├── environment.yml        # Conda environment
├── requirements.txt       # Pip requirements
└── README.md
```

---

## 🧰 Yêu cầu hệ thống

- **Python** ≥ 3.9
- **CUDA-enabled GPU** (khuyến nghị)
- **Conda** (khuyến nghị) hoặc pip
- **PyTorch** + **vLLM**

---

## 🛠 Cài đặt môi trường

### 🔹 Cách 1: Dùng Conda (khuyến nghị)

```bash
conda env create -f environment.yml
conda activate viterbox-vllm
```

### 🔹 Cách 2: Dùng pip

```bash
pip install -r requirements.txt
```

---

## 📥 Tải Model Viterbox

Model được sử dụng trong project:

```
dolly-vn/viterbox
```

**Tải model bằng HuggingFace CLI:**

```bash
huggingface-cli download dolly-vn/viterbox --local-dir /path/to/viterbox
```

&gt; 📌 **Lưu ý:** Ghi nhớ đường dẫn thư mục model sau khi tải xong.

---

## ⚙️ Cấu hình Model cho vLLM

Sau khi tải model, cần chỉ định đường dẫn model local:

1. Mở file `tts.py`
2. Tìm class `ChatterboxTTS`
3. Trong phương thức `from_pretrained`, sửa biến `local_dir`:

```python
local_dir = "/path/to/viterbox"
```

&gt; ➡️ Thay `/path/to/viterbox` bằng đường dẫn thực tế nơi bạn đã tải model.

---

## ▶️ Cách sử dụng

Toàn bộ hướng dẫn sử dụng chi tiết (inference 1 mẫu, batch, cấu hình tham số, v.v.) đã được trình bày trong notebook:

```
test.ipynb
```

👉 Chỉ cần mở notebook và chạy lần lượt các cell.

---

## ⚡ Benchmark

| Metric | Kết quả |
|--------|---------|
| 🚀 Tốc độ suy luận | ~4× nhanh hơn backend thông thường |
| 📉 Overhead | Giảm đáng kể khi batch inference |


---

## 🧩 Mục tiêu của Project

- ✅ Mang vLLM vào pipeline Viterbox/Chatterbox TTS
- ✅ Cải thiện hiệu năng inference cho TTS tiếng Việt
- 🔜 Tạo nền tảng để mở rộng sang:
  - Streaming TTS
  - Multi-speaker
  - Production-grade deployment

---

## ⚠️ Lưu ý quan trọng

| Vấn đề | Giải pháp |
|--------|-----------|
| Repo không bao gồm model weights | Tự tải model từ HuggingFace |
| Lỗi khi chạy vLLM | Kiểm tra phiên bản CUDA, PyTorch, vLLM |

---
