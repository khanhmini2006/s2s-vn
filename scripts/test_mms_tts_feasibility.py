"""Thử nghiệm độc lập model TTS facebook/mms-tts-vie (VITS, Facebook MMS).

Bối cảnh: sau khi gwen-tts-0.6B không đạt realtime (RTF=2.14) và không cải thiện được
trên hạ tầng hiện tại (thiếu CUDA toolkit cho flash-attn, glibc quá cũ cho GGML backend),
chuyển sang thử facebook/mms-tts-vie — theo gợi ý từ handler mẫu trong repo gốc
speech-to-speech (src/speech_to_speech/TTS/facebookmms_handler.py).

Khác biệt so với gwen-tts:
- Dùng transformers.VitsModel chuẩn (tương thích transformers 5.12.1 hiện có, KHÔNG cần
  venv riêng, KHÔNG cần package pip đặc thù).
- Checkpoint nhẹ (~145MB) so với gwen-tts (~1.83GB).
- KHÔNG hỗ trợ voice cloning — chỉ 1 giọng cố định do model quy định.
- License CC-BY-NC-4.0 (chỉ phi thương mại) — khác MIT của gwen-tts.

Đo: thời gian tải model, VRAM sử dụng, RTF (thời gian sinh / độ dài audio), chất lượng nghe thử.
"""
import time
from pathlib import Path

import librosa
import torch
from transformers import AutoTokenizer, VitsModel

MODEL_NAME = "facebook/mms-tts-vie"
TEST_TEXT = "Xin chào, tôi là trợ lý giọng nói tiếng Việt, rất vui được hỗ trợ bạn hôm nay."
OUT_WAV_PATH = Path("/tmp/mms_tts_vie_out.wav")


def vram_mb() -> float:
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024**2)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[test] device={device}")

    print(f"[test] load model {MODEL_NAME} ...")
    t0 = time.time()
    model = VitsModel.from_pretrained(MODEL_NAME).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    load_s = time.time() - t0
    print(f"[test] load xong sau {load_s:.1f}s, VRAM sau load: {vram_mb():.0f} MB")
    print(f"[test] model sampling_rate={model.config.sampling_rate}")

    # warmup
    inputs = tokenizer("xin chào", return_tensors="pt", padding=True)
    with torch.no_grad():
        _ = model(input_ids=inputs.input_ids.to(device), attention_mask=inputs.attention_mask.to(device))

    print("[test] sinh thử câu tiếng Việt ...")
    inputs = tokenizer(TEST_TEXT, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs.input_ids.to(device).long()
    attention_mask = inputs.attention_mask.to(device)

    t0 = time.time()
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)
    gen_s = time.time() - t0
    peak_vram = vram_mb()

    sr = model.config.sampling_rate
    audio_numpy = output.waveform.cpu().numpy().squeeze()
    audio_len_s = len(audio_numpy) / sr

    print(f"[test] sample_rate={sr}, audio_len={audio_len_s:.2f}s, thời gian sinh={gen_s:.2f}s")
    print(f"[test] RTF (gen_time/audio_len) = {gen_s / audio_len_s:.3f}")
    print(f"[test] VRAM đỉnh: {peak_vram:.0f} MB")

    audio_16k = librosa.resample(audio_numpy, orig_sr=sr, target_sr=16000)

    import soundfile as sf

    sf.write(str(OUT_WAV_PATH), audio_16k, 16000)
    print(f"[test] đã ghi file nghe thử (resample 16k): {OUT_WAV_PATH}")
    print("[test] OK — nghe thử file để đánh giá chất lượng tiếng Việt.")


if __name__ == "__main__":
    main()
