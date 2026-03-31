# 🦜 VieNeu-TTS

[![Awesome](https://img.shields.io/badge/Awesome-NLP-green?logo=github)](https://github.com/keon/awesome-nlp)
[![Discord](https://img.shields.io/badge/Discord-Join%20Us-5865F2?logo=discord&logoColor=white)](https://discord.gg/yJt8kzjzWZ)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1V1DjG-KdmurCAhvXrxxTLsa9tteDxSVO?usp=sharing)
[![Hugging Face v2 Turbo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-v2%20Turbo-blue)](https://huggingface.co/pnnbao-ump/VieNeu-TTS-v2-Turbo-GGUF)
[![Hugging Face 0.3B](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-0.3B-orange)](https://huggingface.co/pnnbao-ump/VieNeu-TTS-0.3B)

<img width="1087" height="710" alt="image" src="https://github.com/user-attachments/assets/5534b5db-f30b-4d27-8a35-80f1cf6e5d4d" />

**VieNeu-TTS** là mô hình chuyển đổi văn bản thành giọng nói (TTS) tiếng Việt tiên tiến chạy trên thiết bị, hỗ trợ **clone giọng nói tức thì** và khả năng đọc **song ngữ Anh-Việt**.

> [!IMPORTANT]
> **🚀 VieNeu-TTS-v2 Turbo:** Phiên bản mới nhất được tối ưu hóa cho CPU & Thiết bị cấu hình thấp, hỗ trợ đọc **song ngữ (Code-switching)** mượt mà và tốc độ xử lý cực nhanh.

## ✨ Tính năng nổi bật
- **Song ngữ Anh-Việt**: Chuyển đổi ngôn ngữ tự nhiên và mượt mà nhờ thư viện [sea-g2p](https://github.com/pnnbao97/sea-g2p).
- **Clone giọng nói tức thì**: Clone bất kỳ giọng nói nào chỉ với **3-5 giây** âm thanh mẫu (Chế độ GPU/Standard).
- **Chế độ Turbo siêu nhanh**: Tối ưu hóa cho CPU sử dụng GGUF và ONNX, **KHÔNG cần GPU** và tốn ít RAM.
- **Định danh AI**: Hệ thống đóng dấu bản quyền âm thanh ẩn để đảm bảo trách nhiệm nội dung.
- **Sẵn sàng cho sản xuất**: Tạo âm thanh chất lượng cao 24 kHz, hoạt động hoàn toàn offline.

---

## 🦜 1. Cài đặt & Giao diện Web <a name="installation"></a>

### Thiết lập với `uv` (Khuyến nghị)
`uv` là cách nhanh nhất để quản lý các phụ thuộc. [Cài đặt uv tại đây](https://astral.sh/uv/install).

1. **Clone Repo:**
   ```bash
   git clone https://github.com/pnnbao97/VieNeu-TTS.git
   cd VieNeu-TTS
   ```

2. **Cài đặt các phụ thuộc:**
   - **Lựa chọn 1: Tối giản (Turbo/CPU)** - Nhanh & Nhẹ
     ```bash
     uv sync
     ```
   - **Lựa chọn 2: Đầy đủ (GPU/Standard)** - Chất lượng cao & Cloning
     ```bash
     uv sync --group gpu
     ```

3. **Khởi chạy Giao diện Web:**
   ```bash
   uv run vieneu-web
   ```
   Truy cập giao diện tại `http://127.0.0.1:7860`. Model **Turbo v2** được chọn mặc định để bạn trải nghiệm ngay lập tức.

---

## 📦 2. Sử dụng Python SDK (vieneu) <a name="sdk"></a>

SDK `vieneu` hiện mặc định sử dụng **chế độ Turbo** để đảm bảo tính tương thích cao nhất.

### Bắt đầu nhanh
```bash
pip install vieneu
```

```python
from vieneu import Vieneu

# Khởi tạo chế độ Turbo (Mặc định - Phụ thuộc tối giản)
tts = Vieneu(mode="turbo")

# Tổng hợp giọng nói (sử dụng giọng Nam miền Nam mặc định 'Xuân Vĩnh')
text = "Trước đây, hệ thống điện chủ yếu sử dụng direct current, nhưng Tesla đã chứng minh rằng alternating current is more efficient."
audio = tts.infer(text=text)

# Lưu thành file
tts.save(audio, "output.wav")
print("💾 Đã lưu file output.wav")
```

### Các chế độ nâng cao
| Chế độ | Mô tả | Yêu cầu |
|---|---|---|
| `turbo` | (Mặc định) Siêu nhanh trên CPU | `onnxruntime`, `llama-cpp-python` |
| `remote` | Kết nối tới VieNeu API Server từ xa | `requests` |
| `standard` | Backend PyTorch chất lượng cao | `torch`, `neucodec` |

---

## 🐳 3. Docker & Remote Server <a name="docker-remote"></a>

Triển khai VieNeu-TTS dưới dạng API Server hiệu suất cao (tối ưu cho GPU).

```bash
# Chạy với hỗ trợ GPU
docker run --gpus all -p 23333:23333 pnnbao/vieneu-tts:serve --tunnel
```
Kiểm tra log container để lấy địa chỉ public `bore.pub`.

---

## 🔬 4. Tổng quan mô hình <a name="backbones"></a>

| Model | Định dạng | Thiết bị | Song ngữ | Cloning | Tốc độ |
|---|---|---|---|---|---|
| **VieNeu-v2-Turbo** | GGUF/ONNX | **CPU**/GPU | ✅ | ❌ (Sắp tới) | **Cực hạn** |
| **VieNeu-TTS-v2** | PyTorch | GPU | ✅ | ✅ | **Chuẩn** (Sắp tới) |
| **VieNeu-TTS 0.3B** | PyTorch | GPU/CPU | ❌ | ✅ | **Rất nhanh** |
| **VieNeu-TTS** | PyTorch | GPU/CPU | ❌ | ✅ | **Chuẩn** |

> [!TIP]
> Sử dụng **Turbo v2** cho các ứng dụng trợ lý AI, chatbot, đọc báo trên laptop.
> Sử dụng **GPU/Standard** cho nhu cầu clone giọng nói chất lượng cao và sáng tạo nội dung nghệ thuật.

---

## 🚀 Lộ trình phát triển <a name="roadmap"></a>

- [x] **VieNeu-TTS-v2 Turbo**: Hỗ trợ chuyển đổi ngôn ngữ Anh-Việt (Bilingual).
- [x] **VieNeu-Codec**: Neural codec tối ưu cho tiếng Việt (định dạng ONNX).
- [ ] **VieNeu-TTS-v2 (Non-Turbo)**: Phiên bản đầy đủ chất lượng cao, hỗ trợ **Voice Cloning** và tăng tốc **LMDeploy**.
- [ ] **Turbo Voice Cloning**: Mang tính năng clone giọng nói lên engine Turbo siêu nhẹ.
- [ ] **Mobile SDK**: Hỗ trợ chính thức cho iOS và Android.

---

## 🤝 Hỗ trợ & Liên hệ <a name="support"></a>

- **Hugging Face:** [pnnbao-ump](https://huggingface.co/pnnbao-ump)
- **Discord:** [Tham gia cộng đồng](https://discord.gg/yJt8kzjzWZ)
- **Facebook:** [Pham Nguyen Ngoc Bao](https://www.facebook.com/pnnbao97)
- **Giấy phép:** Apache 2.0 (Sử dụng tự do).

---

**Được thực hiện với ❤️ dành cho cộng đồng TTS Việt Nam**
