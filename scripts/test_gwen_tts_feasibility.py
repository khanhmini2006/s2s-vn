"""Thử nghiệm độc lập model TTS gwen-tts-0.6B (g-group-ai-lab/gwen-tts-0.6B).

Mục đích: xác nhận khả thi trên máy hiện tại TRƯỚC khi viết registry/handler chính thức
(Q5 grilling session — thêm model gwen-tts vào s2s-vn).

Đo: thời gian tải model, VRAM sử dụng, thời gian sinh audio (TTFA-ish), chất lượng nghe thử.
Không đụng vào pipeline chính — script độc lập, xoá được sau khi kết luận.

Cài đặt trước khi chạy:
    pip install qwen-tts soundfile librosa

Ref audio mẫu (giọng yen_nhi) tự tải từ GitHub repo gwen-tts nếu chưa có.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

REF_AUDIO_URL = "https://raw.githubusercontent.com/ggroup-ai-lab/gwen-tts/main/data/ref_audio/yen_nhi.wav"
REF_TEXT = (
    "sao lại không liên quan. các anh lấy vợ rồi các anh cứ đội chị lên đầu làm nóc nhà ấy, "
    "suốt ngày hỏi ý kiến các chị thì làm sao mà ra vấn đề được cho em đúng không."
)
TEST_TEXT = "Xin chào, tôi là trợ lý giọng nói tiếng Việt, rất vui được hỗ trợ bạn hôm nay."

ASSETS_DIR = Path("/tmp/gwen_tts_test_assets")
REF_AUDIO_PATH = ASSETS_DIR / "yen_nhi.wav"
OUT_WAV_PATH = Path("/tmp/gwen_tts_out.wav")


def ensure_ref_audio() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    if not REF_AUDIO_PATH.exists():
        print(f"[test] tải ref audio mẫu -> {REF_AUDIO_PATH}")
        subprocess.run(["curl", "-sL", "-o", str(REF_AUDIO_PATH), REF_AUDIO_URL], check=True)


def vram_mb() -> float:
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024**2)


def main() -> None:
    ensure_ref_audio()

    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

    print("[test] load model g-group-ai-lab/gwen-tts-0.6B (lần đầu sẽ tải checkpoint ~2.5GB) ...")
    t0 = time.time()
    model = Qwen3TTSModel.from_pretrained(
        "g-group-ai-lab/gwen-tts-0.6B",
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )
    load_s = time.time() - t0
    print(f"[test] load xong sau {load_s:.1f}s, VRAM sau load: {vram_mb():.0f} MB")
    print(f"[test] tts_model_type={model.model.tts_model_type}")

    print("[test] sinh thử câu tiếng Việt (voice clone, giọng yen_nhi) ...")
    t0 = time.time()
    wavs, sr = model.generate_voice_clone(
        text=TEST_TEXT,
        language="Vietnamese",
        ref_audio=str(REF_AUDIO_PATH),
        ref_text=REF_TEXT,
        temperature=0.3,
        top_k=20,
        top_p=0.9,
        repetition_penalty=2.0,
        subtalker_temperature=0.1,
        subtalker_top_k=20,
        subtalker_top_p=1.0,
        max_new_tokens=4096,
    )
    gen_s = time.time() - t0
    peak_vram = vram_mb()

    audio_len_s = len(wavs[0]) / sr
    print(f"[test] sample_rate={sr}, audio_len={audio_len_s:.2f}s, thời gian sinh={gen_s:.2f}s")
    print(f"[test] RTF (gen_time/audio_len) = {gen_s / audio_len_s:.2f}")
    print(f"[test] VRAM đỉnh: {peak_vram:.0f} MB")

    import soundfile as sf

    sf.write(str(OUT_WAV_PATH), wavs[0], sr)
    print(f"[test] đã ghi file nghe thử: {OUT_WAV_PATH}")
    print("[test] OK — nghe thử file để đánh giá chất lượng tiếng Việt.")


if __name__ == "__main__":
    main()
