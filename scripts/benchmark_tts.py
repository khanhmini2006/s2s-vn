"""Script benchmark TTS — đo hiệu năng VieNeu-TTS (voice/backend/style khác nhau).

Đo: warmup time, inference time, time-to-first-chunk, audio duration, RTF
(real-time factor = audio_duration / inference_time — RTF < 1 nghĩa là TTS
chạy CHẬM hơn thời gian thực, không đủ cho streaming real-time).

Port từ speech-to-speech/scripts/benchmark_tts.py (bản gốc benchmark nhiều
engine TTS khác nhau: kokoro/qwen3/pocket_tts/chatTTS/facebookMMS). s2s-vn chỉ
có 1 engine TTS (VieNeu-TTS) nên script này benchmark theo biến thể
backend (onnx/pytorch) và voice thay vì theo engine.

Usage:
    python scripts/benchmark_tts.py --text "Xin chào bạn" --iterations 3
    python scripts/benchmark_tts.py --backends onnx pytorch --voices "Trúc Ly"
"""

import argparse
import json
import logging
import time
from queue import Queue
from typing import Any, Dict, List, Optional

import numpy as np

from s2s_vn.pipeline.messages import TTSInput

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_RATE = 16000


class BenchmarkResult:
    """Lưu kết quả benchmark của một biến thể TTS (backend/voice)."""

    def __init__(self, handler_name: str):
        self.handler_name = handler_name
        self.warmup_time = 0.0
        self.inference_times: list[float] = []
        self.time_to_first_chunk: list[float] = []
        self.audio_durations: list[float] = []
        self.errors: list[str] = []

    def add_inference(self, time_taken: float, audio_duration: float, ttfc: Optional[float] = None):
        self.inference_times.append(time_taken)
        self.audio_durations.append(audio_duration)
        if ttfc is not None:
            self.time_to_first_chunk.append(ttfc)

    def add_error(self, error: str):
        self.errors.append(error)

    def get_stats(self) -> Dict[str, Any]:
        if not self.inference_times:
            return {
                "handler": self.handler_name,
                "status": "failed",
                "errors": self.errors,
            }

        avg_time = float(np.mean(self.inference_times))
        avg_audio = float(np.mean(self.audio_durations))
        avg_rtf = avg_audio / avg_time if avg_time > 0 else 0.0

        stats = {
            "handler": self.handler_name,
            "warmup_time": self.warmup_time,
            "avg_inference_time": avg_time,
            "min_inference_time": float(np.min(self.inference_times)),
            "max_inference_time": float(np.max(self.inference_times)),
            "std_inference_time": float(np.std(self.inference_times)),
            "avg_audio_duration": avg_audio,
            "min_audio_duration": float(np.min(self.audio_durations)),
            "max_audio_duration": float(np.max(self.audio_durations)),
            "std_audio_duration": float(np.std(self.audio_durations)),
            "avg_rtf": avg_rtf,
            "total_iterations": len(self.inference_times),
            "errors": self.errors,
        }

        if self.time_to_first_chunk:
            stats["avg_time_to_first_chunk"] = float(np.mean(self.time_to_first_chunk))
            stats["min_time_to_first_chunk"] = float(np.min(self.time_to_first_chunk))
            stats["max_time_to_first_chunk"] = float(np.max(self.time_to_first_chunk))
            stats["std_time_to_first_chunk"] = float(np.std(self.time_to_first_chunk))

        return stats


def benchmark_handler(
    variant_name: str,
    text: str,
    iterations: int,
    backend: str,
    voice: str,
    style: str,
    streaming: bool,
    denoise: bool,
    temperature: float,
) -> BenchmarkResult:
    """Benchmark một biến thể VieNeu-TTS (backend/voice/style)."""
    logger.info(f"Benchmarking {variant_name}...")
    result = BenchmarkResult(variant_name)

    try:
        queue_in: Queue[Any] = Queue()
        queue_out: Queue[Any] = Queue()

        from s2s_vn.TTS.vieneu_tts_handler import VieNeuTTSHandler

        handler = VieNeuTTSHandler(
            queue_in, queue_out,
            voice=voice,
            output_sample_rate=DEFAULT_SAMPLE_RATE,
            streaming=streaming,
            denoise=denoise,
            backend=backend,
            style=style,
            temperature=temperature,
        )

        start_setup = time.perf_counter()
        handler.warmup()
        result.warmup_time = time.perf_counter() - start_setup
        logger.info(f"Handler {variant_name} initialized và warmed up trong {result.warmup_time:.3f}s")

        for i in range(iterations):
            logger.info(f"Iteration {i+1}/{iterations} for {variant_name}")
            start_time = time.perf_counter()
            time_to_first_chunk = None
            first_output = True
            total_bytes = 0

            # process() của VieNeuTTSHandler put trực tiếp AudioOutput vào
            # queue_out (không return list) — đọc lại từ queue sau khi process() xong
            tts_input = TTSInput(text=text, turn_id=i)
            handler.process(tts_input)

            while not queue_out.empty():
                out = queue_out.get_nowait()
                if first_output:
                    time_to_first_chunk = time.perf_counter() - start_time
                    first_output = False
                audio_bytes = getattr(out, "audio", None)
                if audio_bytes:
                    total_bytes += len(audio_bytes)

            end_time = time.perf_counter()
            time_taken = end_time - start_time
            # PCM16 mono → 2 bytes/sample
            audio_duration = (total_bytes / 2) / DEFAULT_SAMPLE_RATE if total_bytes > 0 else 0.0

            result.add_inference(time_taken, audio_duration, time_to_first_chunk)
            ttfc_str = f", TTFC: {time_to_first_chunk:.4f}s" if time_to_first_chunk else ""
            rtf = audio_duration / time_taken if time_taken > 0 else 0
            logger.info(
                f"  Time: {time_taken:.4f}s{ttfc_str}, Audio: {audio_duration:.2f}s, RTF: {rtf:.2f}"
            )

    except Exception as e:
        logger.error(f"Error benchmarking {variant_name}: {e}", exc_info=True)
        result.add_error(str(e))

    return result


def build_benchmark_targets(args) -> List[tuple[str, dict]]:
    """Tạo danh sách (tên biến thể, kwargs) từ tổ hợp backend × voice."""
    targets = []
    for backend in args.backends:
        for voice in args.voices:
            name = f"{backend}[{voice}]"
            targets.append((name, {"backend": backend, "voice": voice}))
    return targets


def print_results(results: List[BenchmarkResult]):
    print("\n" + "=" * 80)
    print("TTS BENCHMARK RESULTS")
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

        print(f"  Avg Audio Duration:   {stats['avg_audio_duration']:.2f}s")
        print(f"  Min Audio Duration:   {stats['min_audio_duration']:.2f}s")
        print(f"  Max Audio Duration:   {stats['max_audio_duration']:.2f}s")
        print(f"  Std Audio Duration:   {stats['std_audio_duration']:.4f}s")
        print(f"  Avg RTF:              {stats['avg_rtf']:.2f}")

        if "avg_time_to_first_chunk" in stats:
            print("\n  Time to First Chunk:")
            print(f"    Avg TTFC:           {stats['avg_time_to_first_chunk']:.4f}s")
            print(f"    Min TTFC:           {stats['min_time_to_first_chunk']:.4f}s")
            print(f"    Max TTFC:           {stats['max_time_to_first_chunk']:.4f}s")
            print(f"    Std TTFC:           {stats['std_time_to_first_chunk']:.4f}s")

        print(f"\n  Total Iterations:     {stats['total_iterations']}")

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
    data = {
        "results": [r.get_stats() for r in results],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark VieNeu-TTS (s2s-vn)")
    parser.add_argument(
        "--text",
        type=str,
        default="Xin chào bạn, đây là bài kiểm tra độ trễ tổng hợp giọng nói.",
        help="Văn bản cần tổng hợp",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["onnx"],
        help="Danh sách backend VieNeu-TTS cần benchmark (onnx | pytorch)",
    )
    parser.add_argument(
        "--voices",
        nargs="+",
        default=["Trúc Ly"],
        help="Danh sách voice cần benchmark",
    )
    parser.add_argument(
        "--style",
        type=str,
        default="tu_nhien",
        help="Style giọng đọc (tu_nhien | tin_tuc | doc_truyen)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Số lần lặp mỗi biến thể (mặc định: 3)",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Tắt streaming (dùng infer() batch thay vì infer_stream())",
    )
    parser.add_argument(
        "--no-denoise",
        action="store_true",
        help="Tắt denoise",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Temperature sinh giọng (mặc định: 0.8)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tts_benchmark_results.json",
        help="File JSON output (mặc định: tts_benchmark_results.json)",
    )

    args = parser.parse_args()

    targets = build_benchmark_targets(args)
    if not targets:
        logger.error("Không có biến thể nào để benchmark (kiểm tra --backends/--voices)")
        return

    results = []
    for variant_name, kwargs in targets:
        result = benchmark_handler(
            variant_name,
            args.text,
            args.iterations,
            backend=kwargs["backend"],
            voice=kwargs["voice"],
            style=args.style,
            streaming=not args.no_streaming,
            denoise=not args.no_denoise,
            temperature=args.temperature,
        )
        results.append(result)

    print_results(results)
    save_results(results, args.output)

    logger.info("TTS benchmarking complete!")


if __name__ == "__main__":
    main()
