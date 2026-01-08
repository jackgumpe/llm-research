"""
DeepSeek-OCR Benchmark Suite
----------------------------
Executes the OCR pipeline across the labelled evaluation set, computing
latency, Character Error Rate (CER), and (optionally) GPU memory usage.

Example:
    python benchmark_suite.py \
        --eval-dir C:/Dev/llm-research/deepseek-ocr/data/eval \
        --ground-truth-dir C:/Dev/llm-research/deepseek-ocr/data/eval/ground_truth \
        --output-path C:/Dev/llm-research/deepseek-ocr/data/benchmarks/run.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

import textdistance
import torch

# Dynamically import run_ocr to reuse its helpers without turning scripts/ into a package
RUN_OCR_PATH = Path(__file__).with_name("run_ocr.py")
spec = importlib.util.spec_from_file_location("deepseek_run_ocr", RUN_OCR_PATH)
run_ocr = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(run_ocr)  # type: ignore


def load_config(config_path: Path) -> Dict:
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_eval_files(eval_dir: Path):
    supported_suffixes = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
    for candidate in sorted(eval_dir.iterdir()):
        if candidate.suffix.lower() in supported_suffixes:
            yield candidate


def load_ground_truth(ground_truth_dir: Path, stem: str) -> Optional[str]:
    candidate = ground_truth_dir / f"{stem}.txt"
    if candidate.exists():
        return candidate.read_text(encoding="utf-8").strip()
    return None


def compute_cer(prediction: str, reference: str) -> float:
    if not reference:
        return 1.0
    distance = textdistance.levenshtein.distance(prediction, reference)
    return distance / max(1, len(reference))


def aggregate_prediction(results: List[Dict]) -> str:
    texts: List[str] = []
    for entry in results:
        out_path = entry.get("output_text_path")
        if out_path and Path(out_path).exists():
            texts.append(Path(out_path).read_text(encoding="utf-8").strip())
    return "\n".join(texts)


def benchmark_quantization(
    quant_level: str,
    config: Dict,
    eval_files: List[Path],
    ground_truth_dir: Path,
    tmp_root: Path,
    device: str,
) -> Dict:
    model_path, adapter_path, resolved_level = run_ocr.get_model_paths(quant_level)
    quant_tmp = tmp_root / f"{resolved_level}_outputs"
    quant_tmp.mkdir(parents=True, exist_ok=True)

    per_file_results = []
    total_wall = 0.0
    vram_peak_mb = 0.0

    for eval_file in eval_files:
        file_start = time.perf_counter()
        images: List[str] = []
        file_tmp_dir = quant_tmp / f"{eval_file.stem}_images"
        if eval_file.suffix.lower() == ".pdf":
            images = run_ocr.process_pdf_for_ocr(str(eval_file), str(file_tmp_dir))
        else:
            file_tmp_dir.mkdir(parents=True, exist_ok=True)
            tmp_image = file_tmp_dir / eval_file.name
            tmp_image.write_bytes(eval_file.read_bytes())
            images = [str(tmp_image)]

        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        ocr_results = run_ocr.run_deepseek_ocr(
            image_paths=images,
            output_dir=str(file_tmp_dir / "ocr"),
            model_path_str=model_path,
            adapter_path_str=adapter_path,
            device=device,
        )

        if device.startswith("cuda") and torch.cuda.is_available():
            vram_peak_mb = max(vram_peak_mb, torch.cuda.max_memory_allocated() / (1024**2))

        prediction = aggregate_prediction(ocr_results)
        ground_truth = load_ground_truth(ground_truth_dir, eval_file.stem)
        cer = None
        notes = ""
        if ground_truth is None:
            notes = "Missing ground truth transcript"
        else:
            cer = compute_cer(prediction, ground_truth)

        elapsed = time.perf_counter() - file_start
        total_wall += elapsed

        per_file_results.append(
            {
                "file": str(eval_file.name),
                "latency_seconds": elapsed,
                "cer": cer,
                "has_ground_truth": ground_truth is not None,
                "notes": notes,
            }
        )

    avg_cer = (
        sum(r["cer"] for r in per_file_results if r["cer"] is not None)
        / max(1, sum(1 for r in per_file_results if r["cer"] is not None))
    )

    return {
        "quantization": resolved_level,
        "model_path": model_path,
        "adapter_path": adapter_path,
        "device": device,
        "total_latency_seconds": total_wall,
        "mean_latency_seconds": total_wall / max(1, len(per_file_results)),
        "vram_peak_mb": vram_peak_mb if vram_peak_mb else None,
        "average_cer": avg_cer,
        "files": per_file_results,
    }


def main():
    parser = argparse.ArgumentParser(description="DeepSeek-OCR benchmark suite.")
    parser.add_argument("--config-path", default="C:/Dev/llm-research/deepseek-ocr/models/config.json")
    parser.add_argument("--eval-dir", default="C:/Dev/llm-research/deepseek-ocr/data/eval")
    parser.add_argument("--ground-truth-dir", default="C:/Dev/llm-research/deepseek-ocr/data/eval/ground_truth")
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument(
        "--quantization-levels",
        default="",
        help="Comma separated list (e.g. fp16,Q5_K_M). Defaults to every entry in config.json.",
    )
    args = parser.parse_args()

    config_path = Path(args.config_path)
    eval_dir = Path(args.eval_dir)
    gt_dir = Path(args.ground_truth_dir)
    output_path = Path(args.output_path) if args.output_path else Path("C:/Dev/llm-research/deepseek-ocr/data/benchmarks") / f"benchmark_{int(time.time())}.json"

    if not eval_dir.exists():
        raise FileNotFoundError(f"Eval directory not found: {eval_dir}")
    if not gt_dir.exists():
        raise FileNotFoundError(f"Ground truth directory not found: {gt_dir}")

    config = load_config(config_path)
    eval_files = list(iter_eval_files(eval_dir))
    if not eval_files:
        raise RuntimeError(f"No evaluation assets found in {eval_dir}")

    if args.quantization_levels:
        quant_levels = [q.strip() for q in args.quantization_levels.split(",") if q.strip()]
    else:
        quant_levels = list(config["models"].keys())

    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_path": str(config_path),
        "eval_dir": str(eval_dir),
        "ground_truth_dir": str(gt_dir),
        "device": args.device,
        "results": [],
    }

    with tempfile.TemporaryDirectory(prefix="ocr_benchmark_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        for quant in quant_levels:
            print(f"[benchmark] Evaluating quantization level: {quant}")
            result = benchmark_quantization(quant, config, eval_files, gt_dir, tmp_root, args.device)
            summary["results"].append(result)

    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[benchmark] Summary written to {output_path}")


if __name__ == "__main__":
    main()
