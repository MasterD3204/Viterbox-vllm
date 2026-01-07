# Viterbox-vLLM 🚀  
**High-performance vLLM backend for Viterbox TTS**

Viterbox-vLLM là phiên bản **tối ưu hiệu năng** của Chatterbox/Viterbox TTS, sử dụng **vLLM** làm backend suy luận.  
Phiên bản này được thiết kế để **tăng tốc độ suy luận lên ~4× so với bản thông thường**, đồng thời hỗ trợ cả **inference đơn mẫu** và **batch inference**.

---

## ✨ Tính năng chính

- ⚡ **Nhanh hơn ~4×** so với backend truyền thống (PyTorch eager)
- 🔥 Sử dụng **vLLM** cho inference hiệu quả và ổn định
- 📦 Hỗ trợ:
  - Inference **1 mẫu**
  - Inference **batch**
- 🧠 Tương thích với model **Viterbox**
- 🧪 Có sẵn notebook hướng dẫn sử dụng (`test.ipynb`)
- 🛠 Dễ dàng tích hợp vào pipeline TTS hiện có

---

## 📂 Cấu trúc repository (tóm tắt)

```text
.
├── src/
│   └── chatterbox_vllm/
│       └── ...
├── tts.py                 # Core TTS implementation
├── test.ipynb             # Notebook hướng dẫn sử dụng
├── environment.yml        # Conda environment
├── requirements.txt       # Pip requirements
└── README.md
🧰 Yêu cầu hệ thống
Python ≥ 3.9

CUDA-enabled GPU (khuyến nghị)

Conda (khuyến nghị) hoặc pip

PyTorch + vLLM

🛠 Cài đặt môi trường
🔹 Cách 1: Dùng Conda (khuyến nghị)
bash
Sao chép mã
conda env create -f environment.yml
conda activate viterbox-vllm
🔹 Cách 2: Dùng pip
bash
Sao chép mã
pip install -r requirements.txt
📥 Tải model Viterbox
Model được sử dụng trong project này là:

bash
Sao chép mã
dolly-vn/viterbox
Bạn có thể tải model bằng HuggingFace CLI hoặc bất kỳ cách nào bạn quen dùng, ví dụ:

bash
Sao chép mã
huggingface-cli download dolly-vn/viterbox --local-dir /path/to/viterbox
📌 Ghi nhớ đường dẫn thư mục model sau khi tải xong.

⚙️ Cấu hình model cho vLLM
Sau khi tải model xong, bạn cần chỉ định đường dẫn model local cho code.

Bước 1: Mở file tts.py
Bước 2: Tìm class ChatterboxTTS
Bước 3: Trong phương thức from_pretrained, sửa biến local_dir:
python
Sao chép mã
local_dir = "/path/to/viterbox"
➡️ Thay /path/to/viterbox bằng đường dẫn thực tế nơi bạn đã tải model về.

▶️ Cách sử dụng
📓 Toàn bộ hướng dẫn sử dụng chi tiết (inference 1 mẫu, batch, cấu hình tham số, v.v.)
đã được trình bày trong notebook:

text
Sao chép mã
test.ipynb
👉 Chỉ cần mở notebook và chạy lần lượt các cell.

⚡ Benchmark (tham khảo)
🚀 Tốc độ suy luận: ~4× nhanh hơn backend thông thường

📉 Giảm overhead khi batch inference

💡 Phù hợp cho:

Research

Demo

Production inference

(Kết quả benchmark phụ thuộc GPU và batch size)

🧩 Mục tiêu của project
Mang vLLM vào pipeline Viterbox / Chatterbox TTS

Cải thiện hiệu năng inference cho TTS tiếng Việt

Tạo nền tảng để mở rộng sang:

Streaming TTS

Multi-speaker

Production-grade deployment

📌 Ghi chú
Repo không bao gồm model weights

Người dùng cần tự tải model từ HuggingFace

Nếu bạn gặp lỗi khi chạy vLLM, hãy kiểm tra:

Phiên bản CUDA

Phiên bản PyTorch

Phiên bản vLLM
