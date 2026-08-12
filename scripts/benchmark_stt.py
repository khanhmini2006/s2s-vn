"""Script benchmark STT — so sánh hiệu năng các STT handler tiếng Việt.

Đo: thời gian inference, warmup, time-to-first-token, chất lượng transcription.
Port từ speech-to-speech/scripts/benchmark_stt.py (bản gốc benchmark
whisper/whisper-mlx/mlx-audio-whisper/faster-whisper/parakeet-tdt), rút gọn
theo 2 STT backend hiện có của s2s-vn: phowhisper-medium (faster-whisper CT2)
và zipformer-vi-6000h (sherpa-onnx).

Usage:
    python scripts/benchmark_stt.py --audio_file path/to/audio.wav --iterations 10
    python scripts/benchmark_stt.py --audio_file path/to/audio.wav --handlers phowhisper-medium
"""

import argparse
import json
import logging
import time
from pathlib import Path
from queue import Queue
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf

from s2s_vn.pipeline.messages import AudioMode, VADAudio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class BenchmarkResult:
    """Lưu kết quả benchmark của một STT handler."""

    def __init__(self, handler_name: str):
        self.handler_name = handler_name
        self.warmup_time = 0.0
        self.inference_times: list[float] = []
        self.time_to_first_token: list[float] = []
        self.transcriptions: list[str] = []
        self.errors: list[str] = []

    def add_inference(self, time_taken: float, transcription: Any, ttft: Optional[float] = None):
        self.inference_times.append(time_taken)
        self.transcriptions.append(transcription)
        if ttft is not None:
            self.time_to_first_token.append(ttft)

    def add_error(self, error: str):
        self.errors.append(error)

    def get_stats(self) -> Dict[str, Any]:
        """Tính thống kê từ kết quả benchmark."""
        if not self.inference_times:
            return {
                "handler": self.handler_name,
                "status": "failed",
                "errors": self.errors,
            }

        stats = {
            "handler": self.handler_name,
            "warmup_time": self.warmup_time,
            "avg_inference_time": float(np.mean(self.inference_times)),
            "min_inference_time": float(np.min(self.inference_times)),
            "max_inference_time": float(np.max(self.inference_times)),
            "std_inference_time": float(np.std(self.inference_times)),
            "total_iterations": len(self.inference_times),
            "errors": self.errors,
            "sample_transcription": self.transcriptions[0] if self.transcriptions else None,
        }

        if self.time_to_first_token:
            stats["avg_time_to_first_token"] = float(np.mean(self.time_to_first_token))
            stats["min_time_to_first_token"] = float(np.min(self.time_to_first_token))
            stats["max_time_to_first_token"] = float(np.max(self.time_to_first_token))
            stats["std_time_to_first_token"] = float(np.std(self.time_to_first_token))

        return stats


def load_audio(audio_path: str) -> np.ndarray:
    """Đọc file audio, trả về numpy array float32 16kHz mono."""
    logger.info(f"Loading audio from: {audio_path}")
    audio, sample_rate = sf.read(audio_path)

    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    if sample_rate != 16000:
        logger.warning(f"Audio sample rate is {sample_rate}Hz, resampling to 16000Hz")
        from scipy.signal import resample_poly

        audio = resample_poly(audio, 16000, sample_rate)

    return audio.astype(np.float32)


def _audio_to_pcm16(audio: np.ndarray) -> bytes:
    """float32 [-1, 1] → PCM16 bytes (VADAudio.audio là PCM16 theo pipeline/messages.py)."""
    return (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()


def benchmark_handler(
    handler_name: str,
    audio: np.ndarray,
    iterations: int,
) -> BenchmarkResult:
    """Benchmark một STT handler."""
    logger.info(f"Benchmarking {handler_name}...")
    result = BenchmarkResult(handler_name)
    pcm16 = _audio_to_pcm16(audio)

    try:
        queue_in: Queue[Any] = Queue()
        queue_out: Queue[Any] = Queue()

        handler: Any = None
        if handler_name == "phowhisper-medium":
            from s2s_vn.STT.whisper_stt_handler import WhisperSTTHandler

            handler = WhisperSTTHandler(
                queue_in, queue_out,
                enable_live_transcription=False,
            )
        elif handler_name == "zipformer-vi-6000h":
            from s2s_vn.STT.zipformer_stt_handler import ZipformerSTTHandler

            handler = ZipformerSTTHandler(
                queue_in, queue_out,
                enable_live_transcription=False,
            )
        else:
            raise ValueError(f"Unknown handler: {handler_name}")

        # warmup() tải model — tính riêng thời gian này, tách khỏi vòng lặp inference
        start_warmup = time.perf_counter()
        handler.warmup()
        result.warmup_time = time.perf_counter() - start_warmup
        logger.info(f"Handler {handler_name} initialized và warmed up trong {result.warmup_time:.3f}s")

        # Warmup thêm trên audio thật (loại khỏi timing)
        handler.process(VADAudio(audio=pcm16, mode=AudioMode.FINAL, turn_id=-1))

        for i in range(iterations):
            logger.info(f"Iteration {i+1}/{iterations} for {handler_name}")

            start_time = time.perf_counter()
            transcription_obj = handler.process(
                VADAudio(audio=pcm16, mode=AudioMode.FINAL, turn_id=i)
            )
            end_time = time.perf_counter()

            # WhisperSTTHandler/ZipformerSTTHandler trả Transcription | None
            # (không streaming từng token → không có TTFT thật, coi TTFT = tổng thời gian)
            transcription = transcription_obj.text if transcription_obj else None
            time_taken = end_time - start_time
            result.add_inference(time_taken, transcription, ttft=time_taken)

            text_preview = str(transcription)[:50] if transcription is not None else "(none)"
            logger.info(f"  Time: {time_taken:.4f}s, Text: {text_preview}...")

    except Exception as e:
        logger.error(f"Error benchmarking {handler_name}: {e}", exc_info=True)
        result.add_error(str(e))

    return result


def print_results(results: List[BenchmarkResult]):
    """In kết quả benchmark dạng bảng."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    for result in results:
        stats = result.get_stats()
        print(f"\nHandler: {stats['handler']}")
        print("-" * 80)

        if stats.get("status") == "failed":
            print("  Status: FAILED")
            print(f"  Errors: {stats['errors']}")
            continue

        print(f"  Warmup Time:          {stats['warmup_time']:.4f}s")
        print(f"  Avg Inference Time:   {stats['avg_inference_time']:.4f}s")
        print(f"  Min Inference Time:   {stats['min_inference_time']:.4f}s")
        print(f"  Max Inference Time:   {stats['max_inference_time']:.4f}s")
        print(f"  Std Deviation:        {stats['std_inference_time']:.4f}s")

        if "avg_time_to_first_token" in stats:
            print("\n  Time to First Token:")
            print(f"    Avg TTFT:           {stats['avg_time_to_first_token']:.4f}s")
            print(f"    Min TTFT:           {stats['min_time_to_first_token']:.4f}s")
            print(f"    Max TTFT:           {stats['max_time_to_first_token']:.4f}s")
            print(f"    Std TTFT:           {stats['std_time_to_first_token']:.4f}s")

        print(f"\n  Total Iterations:     {stats['total_iterations']}")
        print(f"  Sample Transcription: {stats['sample_transcription']}")

        if stats["errors"]:
            print(f"  Errors: {stats['errors']}")

    print("\n" + "=" * 80)
    print("COMPARISON (Average Inference Time)")
    print("=" * 80)

    successful_results = [r for r in results if r.inference_times]
    if successful_results:
        sorted_results = sorted(successful_results, key=lambda x: np.mean(x.inference_times))

        fastest = sorted_results[0]
        fastest_time = np.mean(fastest.inference_times)

        for result in sorted_results:
            avg_time = np.mean(result.inference_times)
            speedup = avg_time / fastest_time
            print(f"  {result.handler_name:25s}: {avg_time:.4f}s  ({speedup:.2f}x slower than fastest)")


def save_results(results: List[BenchmarkResult], output_file: str):
    """Lưu kết quả benchmark ra file JSON."""
    data = {
        "results": [r.get_stats() for r in results],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark STT handlers (s2s-vn)")
    parser.add_argument(
        "--audio_file",
        type=str,
        required=True,
        help="Đường dẫn file audio để benchmark",
    )
    parser.add_argument(
        "--handlers",
        nargs="+",
        default=["phowhisper-medium", "zipformer-vi-6000h"],
        help="Danh sách handler cần benchmark (mặc định: cả 2)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Số lần lặp mỗi handler (mặc định: 5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="stt_benchmark_results.json",
        help="File JSON output (mặc định: stt_benchmark_results.json)",
    )

    args = parser.parse_args()

    if not Path(args.audio_file).exists():
        logger.error(f"Audio file not found: {args.audio_file}")
        return

    audio = load_audio(args.audio_file)
    logger.info(f"Audio loaded: {len(audio)} samples, {len(audio)/16000:.2f}s duration")

    results = []
    for handler_name in args.handlers:
        result = benchmark_handler(handler_name, audio, args.iterations)
        results.append(result)

    print_results(results)
    save_results(results, args.output)

    logger.info("Benchmarking complete!")


if __name__ == "__main__":
    main()
