import re

# Pre-compiled regex (from core_utils)
_RE_NEWLINE = re.compile(r"[\r\n]+")
_RE_SENTENCE_END = re.compile(r"(?<=[.!?\u2026])\s+")
_RE_MINOR_PUNCT = re.compile(r"(?<=[,;:\-\u2013\u2014])\s+")
import time
import numpy as np
import soundfile as sf
import onnxruntime as ort
import gradio as gr
from sea_g2p import SEAPipeline
from llama_cpp import Llama
from concurrent.futures import ThreadPoolExecutor
import json
import os

# ──────────────────────────────────────────────────────────────────────────
# Configuration & Constants
# ──────────────────────────────────────────────────────────────────────────
GGUF_FILENAME = "vieneu-tts-v2-turbo-test.gguf"
DECODER_FILENAME = "vieneu_decoder.onnx"

# ──────────────────────────────────────────────────────────────────────────
# Speaker Embeddings (Pre-calculated: 30 Dataset Spks + Mai)
# ──────────────────────────────────────────────────────────────────────────
SPEAKER_EMBEDDINGS_FILE = "speaker_embeddings.json"
if os.path.exists(SPEAKER_EMBEDDINGS_FILE):
    with open(SPEAKER_EMBEDDINGS_FILE, "r") as f:
        SPEAKER_EMBEDDINGS = json.load(f)
else:
    SPEAKER_EMBEDDINGS = {}
    print(f"⚠️ Warning: {SPEAKER_EMBEDDINGS_FILE} not found.")

SPEAKER_LIST = list(SPEAKER_EMBEDDINGS.keys())
if 'mai' in SPEAKER_LIST:
    SPEAKER_LIST.remove('mai')
    SPEAKER_LIST = ['mai'] + SPEAKER_LIST
BUILTIN_VOICE_IDS = {
    "doan-trang": 0,
    "thuc-doan": 1,
    "pham-tuyen": 2,
    "xuan-vinh": 3,
}
SPEAKER_LIST = list(BUILTIN_VOICE_IDS.keys()) + SPEAKER_LIST

g2p = SEAPipeline(lang="vi")

VOICE_ID_MAP = {
    "doan-trang": 0, "thuc-doan": 1, "pham-tuyen": 2, "xuan-vinh": 3,
    "spk_00083": 0,  "spk_00018": 1, "spk_00033": 2,  "spk_00001": 3,
}

# ──────────────────────────────────────────────────────────────────────────
# Chunking
# ──────────────────────────────────────────────────────────────────────────
def split_into_chunks(full_phones: str, max_chunk_size: int = 256, min_chunk_size: int = 10) -> list[str]:
    """
    Chiến lược:
    1. Ưu tiên tách tối đa: Mỗi câu (.?!) là một chunk riêng.
    2. Nếu câu quá dài (>max), tách tiếp theo dấu phụ (,:;-) hoặc space.
    3. Cuối cùng mới gộp các chunk cực ngắn (<10) vào nhau để tránh bị cụt.
    """
    if not full_phones:
        return []

    # Bước 1: Tách theo đoạn (newline)
    paragraphs = _RE_NEWLINE.split(full_phones.strip())
    raw_parts: list[str] = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # Bước 2: Luôn tách theo câu (.?!)
        sentences = _RE_SENTENCE_END.split(para)
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            
            # Nếu câu quá dài, phải xé nhỏ tiếp
            if len(sent) > max_chunk_size:
                sub_parts = _RE_MINOR_PUNCT.split(sent)
                sub_buf = ""
                for part in sub_parts:
                    part = part.strip()
                    if not part:
                        continue

                    if len(part) > max_chunk_size:
                        # Flush sub_buf trước
                        if sub_buf:
                            raw_parts.append(sub_buf)
                            sub_buf = ""
                        # Xé theo space
                        words = part.split()
                        current = ""
                        for word in words:
                            if current and len(current) + 1 + len(word) > max_chunk_size:
                                raw_parts.append(current)
                                current = word
                            else:
                                current = (current + " " + word) if current else word
                        if current:
                            raw_parts.append(current)
                    elif sub_buf and len(sub_buf) + 1 + len(part) > max_chunk_size:
                        # sub_buf đầy, flush rồi bắt đầu mới
                        raw_parts.append(sub_buf)
                        sub_buf = part
                    else:
                        sub_buf = (sub_buf + " " + part) if sub_buf else part

                if sub_buf:
                    raw_parts.append(sub_buf)
            else:
                raw_parts.append(sent)

    # Bước 3: Gộp các chunk quá ngắn (< min_chunk_size)
    if not raw_parts:
        return []

    merged: list[str] = []
    i = 0
    while i < len(raw_parts):
        current = raw_parts[i]
        
        # Nếu chunk hiện tại quá ngắn, cố gắng gộp với chunk kế tiếp
        while len(current) < min_chunk_size and i + 1 < len(raw_parts):
            next_p = raw_parts[i + 1]
            # Chỉ gộp nếu không vượt quá giới hạn tối đa
            if len(current) + 1 + len(next_p) <= max_chunk_size:
                current = (current + " " + next_p).strip()
                i += 1
            else:
                break
        
        merged.append(current)
        i += 1

    # Xử lý trường hợp chunk cuối cùng vẫn quá ngắn → gộp ngược vào trước
    if len(merged) >= 2 and len(merged[-1]) < min_chunk_size:
        last = merged.pop()
        if len(merged[-1]) + 1 + len(last) <= max_chunk_size:
            merged[-1] = (merged[-1] + " " + last).strip()
        else:
            merged.append(last)

    return [m for m in merged if m]


def get_silence_duration(chunk_phones: str) -> float:
    stripped = chunk_phones.strip()
    if re.search(r'[.!?]$', stripped):
        return 0.3
    elif re.search(r'[,;-]$', stripped):
        return 0.15
    return 0.05


# ──────────────────────────────────────────────────────────────────────────
# Inference Model
# ──────────────────────────────────────────────────────────────────────────
class InferenceModel:
    def __init__(self):
        self.llm = None
        self.decoder_sess = None
        self.sample_rate = 24000

    def load(self):
        print("🚀 Loading models locally...")
        self.llm = Llama(
            model_path=GGUF_FILENAME,
            n_ctx=4096,
            n_gpu_layers=-1,
            flash_attn=True,
            verbose=False,
            repeat_penalty=1.15,
        )
        self.decoder_sess = ort.InferenceSession(
            DECODER_FILENAME, providers=["CPUExecutionProvider"]
        )
        print("✅ Models loaded successfully!")

    def decode_audio(self, ids, voice_id=-1, embedding=None):
        tokens = np.array(ids, dtype=np.int64)[None, :]
        v_id = np.array([voice_id], dtype=np.int64)
        if embedding is None:
            embedding = np.zeros((1, 128), dtype=np.float32)
        elif embedding.ndim == 1:
            embedding = embedding[None, :]
        inputs = {"content_ids": tokens, "voice_id": v_id, "evoice_embedding": embedding}
        audio = self.decoder_sess.run(None, inputs)[0]
        return audio[0]

    def _prepare_embedding(self, speaker_id: str):
        """Trả về (voice_id, embedding) từ speaker_id."""
        voice_id = VOICE_ID_MAP.get(speaker_id, -1)
        val = SPEAKER_EMBEDDINGS.get(speaker_id)
        emb = np.array(val, dtype=np.float32) if val is not None else None
        if emb is not None and emb.ndim == 1:
            emb = emb[None, :]
        return voice_id, emb

    def _build_prompt(self, chunk: str) -> str:
        return (
            f"<|speaker_16|>"
            f"<|TEXT_PROMPT_START|>{chunk}<|TEXT_PROMPT_END|>"
            f"<|SPEECH_GENERATION_START|>"
        )

    def _llm_generate(self, chunk: str, temperature: float, top_k: int, reset_cache: bool = True) -> list[int]:
        """Chạy LLM cho một chunk, trả về danh sách speech token IDs.
        
        reset_cache=True: reset KV cache trước khi generate — bắt buộc dùng
        trong stream mode để tránh model bị nhiễu context từ chunk trước.
        Batch mode không cần vì ThreadPoolExecutor chạy song song, không
        có KV cache chung giữa các thread.
        """
        if reset_cache:
            self.llm.reset()

        result = self.llm(
            self._build_prompt(chunk),
            max_tokens=2048,      # Tăng để tránh bị cắt cụt chunk dài
            temperature=temperature,
            top_k=50,
            top_p=0.95,           # Lọc bớt các token xác suất quá thấp
            min_p=0.05,           # Giúp output ổn định và rõ ràng hơn
            stop=["<|SPEECH_GENERATION_END|>"],
            repeat_penalty=1.15,  # Chống lặp từ/kéo dài âm vô nghĩa
            echo=False,
        )
        generated = result["choices"][0]["text"]
        return [int(x) for x in re.findall(r"<\|speech_(\d+)\|>", generated)]

    # ── Generate all (batch mode, parallel decode) ─────────────────────
    def generate_tokens(self, text, speaker_id, temperature=0.7, top_k=50):
        voice_id, emb = self._prepare_embedding(speaker_id)
        full_phones = g2p.run(text)
        chunks = split_into_chunks(full_phones)

        all_results = []
        start_gen = time.time()

        for i, chunk in enumerate(chunks):
            if len(chunks) > 1:
                print(f"🎙️ Processing chunk {i+1}/{len(chunks)} ({len(chunk)} chars)...")
            ids = self._llm_generate(chunk, temperature, top_k)
            if ids:
                all_results.append((chunk, ids))

        gen_time = time.time() - start_gen

        if not all_results:
            return None, 0, len(full_phones), 0, 0

        start_decode = time.time()
        with ThreadPoolExecutor() as executor:
            all_wavs = list(
                executor.map(lambda r: self.decode_audio(r[1], voice_id, emb), all_results)
            )
        decode_time = time.time() - start_decode

        final_wav = []
        for i, (wav, (chunk_phones, _)) in enumerate(zip(all_wavs, all_results)):
            final_wav.append(wav)
            if i < len(all_wavs) - 1:
                silence_dur = get_silence_duration(chunk_phones)
                final_wav.append(np.zeros(int(self.sample_rate * silence_dur), dtype=np.float32))

        total_tokens = sum(len(r[1]) for r in all_results)
        return np.concatenate(final_wav), total_tokens, len(full_phones), gen_time, decode_time

    # ── Stream mode: yield từng chunk ngay sau khi xong ───────────────
    def stream_tokens(self, text, speaker_id, temperature=0.7, top_k=50):
        """
        Generator: yield (sample_rate, wav_chunk) cho từng chunk.
        Chunk đầu tiên được yield sớm nhất có thể để giảm perceived latency.
        """
        voice_id, emb = self._prepare_embedding(speaker_id)
        full_phones = g2p.run(text)
        chunks = split_into_chunks(full_phones)

        first = True
        for i, chunk in enumerate(chunks):
            if len(chunks) > 1:
                print(f"🎙️ Streaming chunk {i+1}/{len(chunks)}...")

            # reset_cache=True: xóa KV cache trước mỗi chunk để tránh
            # model bị nhiễu context cũ → hết lặp nội dung giữa các chunk
            ids = self._llm_generate(chunk, temperature, top_k, reset_cache=True)
            if not ids:
                continue

            wav = self.decode_audio(ids, voice_id, emb)

            yield self.sample_rate, wav

            # Trick từ Kokoro: yield một frame silence sau chunk đầu tiên
            # để Gradio streaming player kích hoạt ngay lập tức
            if first:
                first = False
                yield self.sample_rate, np.zeros(1, dtype=np.float32)

            # Thêm silence giữa các chunk (trừ chunk cuối)
            if i < len(chunks) - 1:
                silence_dur = get_silence_duration(chunk)
                silence = np.zeros(int(self.sample_rate * silence_dur), dtype=np.float32)
                yield self.sample_rate, silence


# ──────────────────────────────────────────────────────────────────────────
# Global Model Instance
# ──────────────────────────────────────────────────────────────────────────
model = InferenceModel()


# ──────────────────────────────────────────────────────────────────────────
# Gradio Handler Functions
# ──────────────────────────────────────────────────────────────────────────
def tts_predict(text, speaker_id, temperature, top_k):
    """Batch mode: generate toàn bộ rồi trả về."""
    if not text:
        return None, "Vui lòng nhập văn bản."
    if not speaker_id:
        return None, "Vui lòng chọn giọng nói."

    try:
        start_time = time.time()
        if model.llm is None:
            model.load()

        val = SPEAKER_EMBEDDINGS.get(speaker_id)
        if val is None and speaker_id not in BUILTIN_VOICE_IDS:
            return None, f"⚠️ Lỗi: Không tìm thấy embedding cho '{speaker_id}'."

        wav, num_tokens, phone_len, gen_time, decode_time = model.generate_tokens(
            text, speaker_id, temperature, top_k
        )
        if wav is None:
            return None, "Không thể tạo mã âm thanh. Hãy thử lại với văn bản khác."

        duration = time.time() - start_time
        audio_len = len(wav) / model.sample_rate
        rtf = duration / audio_len if audio_len > 0 else 0

        info = (
            f"⚡ Thành công!\n"
            f"- Giọng nói: {speaker_id}\n"
            f"- Tổng thời gian: {duration:.2f}s (RTF: {rtf:.3f})\n"
            f"- Thời gian Gen (LLM): {gen_time:.2f}s\n"
            f"- Thời gian Decode (Parallel): {decode_time:.2f}s\n"
            f"- Số token: {num_tokens}"
        )

        output_path = "output.wav"
        sf.write(output_path, wav, model.sample_rate)
        return output_path, info

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"⚠️ Lỗi: {str(e)}"


def tts_stream(text, speaker_id, temperature, top_k):
    """Stream mode: yield từng chunk, người dùng nghe ngay chunk đầu."""
    if not text:
        return
    if not speaker_id:
        return

    if model.llm is None:
        model.load()

    val = SPEAKER_EMBEDDINGS.get(speaker_id)
    if val is None and speaker_id not in BUILTIN_VOICE_IDS:
        return

    yield from model.stream_tokens(text, speaker_id, temperature, top_k)


# ──────────────────────────────────────────────────────────────────────────
# Gradio UI
# ──────────────────────────────────────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Soft(), title="VieNeu-TTS") as demo:
    gr.HTML("""
        <div style="text-align: center; max-width: 800px; margin: 0 auto; padding: 20px;">
            <h1 style="font-weight: 900; letter-spacing: -0.02em; margin-bottom: 0.5rem; color: #1f2937; font-size: 2.5rem;">
                🦜 VieNeu-TTS Turbo <span style="color: #3b82f6;">GGUF</span>
            </h1>
            <p style="color: #4b5563; font-size: 1.1rem;">
                Mô hình TTS Tiếng Việt siêu nhanh, hỗ trợ đa giọng nói với chất lượng cao.
            </p>
        </div>
    """)

    with gr.Row():
        # ── Left column: inputs ────────────────────────────────────────
        with gr.Column(scale=3):
            input_text = gr.Textbox(
                label="Văn bản muốn đọc",
                placeholder="Nhập nội dung Tiếng Việt...",
                lines=8,
            )
            speaker_dropdown = gr.Dropdown(
                choices=SPEAKER_LIST,
                value=SPEAKER_LIST[0] if SPEAKER_LIST else None,
                label="Chọn giọng nói (Speaker)",
                info="Bao gồm giọng Mai và 30 giọng khác từ dataset."
            )
            with gr.Accordion("Tùy chọn nâng cao", open=False):
                with gr.Row():
                    temp_slider = gr.Slider(0.1, 1.5, value=0.7, step=0.1, label="Temperature")
                    topk_slider = gr.Slider(1, 100, value=50, step=1, label="Top-K")

        # ── Right column: outputs (tabbed Generate / Stream) ──────────
        with gr.Column(scale=2):
            with gr.Tabs():
                # ── Tab 1: Batch Generate ──────────────────────────────
                with gr.Tab("🎵 Generate"):
                    out_audio = gr.Audio(
                        label="Kết quả (Generated Speech)",
                        type="filepath",
                        interactive=False,
                        autoplay=True,
                    )
                    generate_btn = gr.Button("🎤 Bắt đầu tạo giọng nói", variant="primary", size="lg")
                    out_info = gr.Textbox(label="Thông tin hệ thống", lines=8)

                # ── Tab 2: Stream ──────────────────────────────────────
                with gr.Tab("⚡ Stream"):
                    gr.Markdown(
                        "**Phát âm thanh ngay từ chunk đầu tiên** — không cần chờ toàn bộ văn bản được xử lý."
                    )
                    out_stream = gr.Audio(
                        label="Stream Output",
                        streaming=True,
                        autoplay=True,
                        interactive=False,
                    )
                    with gr.Row():
                        stream_btn = gr.Button("▶ Bắt đầu Stream", variant="primary", size="lg")
                        stop_btn  = gr.Button("■ Dừng", variant="stop")

    # ── Event bindings ─────────────────────────────────────────────────
    generate_btn.click(
        fn=tts_predict,
        inputs=[input_text, speaker_dropdown, temp_slider, topk_slider],
        outputs=[out_audio, out_info],
    )

    stream_event = stream_btn.click(
        fn=tts_stream,
        inputs=[input_text, speaker_dropdown, temp_slider, topk_slider],
        outputs=[out_stream],
    )
    stop_btn.click(fn=None, cancels=stream_event)

    # ── Examples ───────────────────────────────────────────────────────
    gr.Examples(
        examples=[
            [
                "Nắng chiều hanh hao hắt qua ô cửa sổ cũ kỹ, đổ dài trên mặt bàn gỗ đã bạc màu thời gian. "
                "Bà cụ Tư ngồi đó, đôi bàn tay gầy guộc run rẩy lật giở từng trang album ảnh ố vàng. "
                "Khói bếp nhà ai phía xa xa bay lên, quyện vào làn sương mờ ảo của buổi hoàng hôn vùng cao.",
                "mai", 0.7, 50,
            ],
            [
                "Việt Nam đất nước ta ơi, mênh mông biển lúa đâu trời đẹp hơn. "
                "Cánh cò bay lả dập dờn, mây mờ che đỉnh Trường Sơn sớm chiều.",
                SPEAKER_LIST[1] if len(SPEAKER_LIST) > 1 else (SPEAKER_LIST[0] if SPEAKER_LIST else ""),
                0.8, 50,
            ],
        ],
        inputs=[input_text, speaker_dropdown, temp_slider, topk_slider],
    )


if __name__ == "__main__":
    demo.launch()