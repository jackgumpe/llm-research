import argparse
import json
import logging
import os
import shutil
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import fitz  # PyMuPDF
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths / Constants
ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_FILE_PATH = ROOT_DIR / "models" / "config.json"
PROGRESS_LOG_PATH = ROOT_DIR / "data" / "ocr_progress.json"
SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"}
DEFAULT_PROMPT = "<image>\nFree OCR."
_ORIGINAL_TORCH_AUTOCAST = torch.autocast


def _safe_autocast(device_type, *args, **kwargs):
    """
    Wrap torch.autocast so bf16 requests on unsupported GPUs fall back to fp16.
    The DeepSeek infer path always asks for bfloat16, which Turing cards lack.
    """
    dtype = kwargs.get("dtype")
    if device_type == "cuda" and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        kwargs["dtype"] = torch.float16
    return _ORIGINAL_TORCH_AUTOCAST(device_type, *args, **kwargs)


@contextmanager
def _patch_autocast():
    original = torch.autocast
    torch.autocast = _safe_autocast
    try:
        yield
    finally:
        torch.autocast = original


def load_config() -> Dict:
    if not CONFIG_FILE_PATH.exists():
        logger.error(f"Configuration file not found: {CONFIG_FILE_PATH}. Please ensure it exists.")
        raise FileNotFoundError(f"Config file missing: {CONFIG_FILE_PATH}")
    with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


CONFIG = load_config()


def _resolve_repo_or_path(local_path: str, repo_id: Optional[str], label: str) -> Union[str, Path]:
    """
    Return a usable source for model/adapter loading.
    Prefers local path; falls back to repo_id if provided.
    """
    if local_path:
        path_obj = Path(local_path)
        if path_obj.exists():
            return path_obj
        logger.warning(f"{label} not found locally at {path_obj}.")
    if repo_id:
        logger.info(f"Falling back to Hugging Face repo '{repo_id}' for {label}.")
        return repo_id
    raise ValueError(f"No valid source available for {label}. Please download files or set repo id.")


def get_model_sources(quantization_level: str):
    model_config = CONFIG["models"].get(quantization_level)
    if not model_config:
        logger.warning(f"Quantization level '{quantization_level}' not found. Falling back to default.")
        model_config = CONFIG["models"].get(CONFIG["default_quantization"])
        if not model_config:
            logger.warning(f"Default quantization '{CONFIG['default_quantization']}' missing. Falling back to 'fp16'.")
            model_config = CONFIG["models"].get("fp16")

    if not model_config:
        raise ValueError("No valid model configuration found. Please update config.json.")

    model_source = _resolve_repo_or_path(
        model_config.get("model_path", ""),
        model_config.get("huggingface_repo_id"),
        "model",
    )
    adapter_source = None
    adapter_path = model_config.get("adapter_path")
    adapter_repo = model_config.get("adapter_repo_id")
    if adapter_path or adapter_repo:
        try:
            adapter_source = _resolve_repo_or_path(adapter_path or "", adapter_repo, "adapter")
        except ValueError:
            adapter_source = None
            logger.warning("Adapter source unavailable; proceeding without adapter.")

    return model_source, adapter_source, model_config["quantization_level"]


def _select_dtype(device: str) -> torch.dtype:
    """
    Returns the dtype to use for DeepSeek-OCR weights. The upstream `infer`
    path currently assumes CUDA tensors, so we enforce GPU availability.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device detected. The current DeepSeek-OCR release requires an NVIDIA GPU.")

    if device.lower() != "cuda":
        logger.warning("DeepSeek-OCR currently forces CUDA tensors internally; switching device to CUDA.")

    major, _ = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float16


def _load_tokenizer_and_model(
    model_source: Union[str, Path],
    device: str,
    quantization_level: str,
) -> Tuple[AutoTokenizer, AutoModel]:
    quant_level = (quantization_level or "").lower()
    dtype = _select_dtype(device)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
    except Exception as err:
        logger.critical("Failed to load tokenizer from %s: %s", model_source, err, exc_info=True)
        raise RuntimeError(f"Tokenizer loading failed: {err}") from err

    model_kwargs = {
        "trust_remote_code": True,
        "use_safetensors": True,
        "low_cpu_mem_usage": True,
        "device_map": "auto",
    }
    if quant_level != "fp16":
        logger.info("Applying 4-bit quantization settings for level '%s'.", quantization_level)
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model_kwargs["quantization_config"] = quant_config
    else:
        model_kwargs["torch_dtype"] = dtype

    try:
        logger.info("Loading model from %s using accelerate device_map=auto.", model_source)
        model = AutoModel.from_pretrained(model_source, **model_kwargs)
        model.eval()
    except Exception as err:
        logger.critical("Failed to load DeepSeek-OCR model from %s: %s", model_source, err, exc_info=True)
        raise RuntimeError(f"Model loading failed: {err}") from err

    return tokenizer, model


import concurrent.futures

def _save_ocr_result(output_filepath: str, text: str):
    """Helper to save text to disk asynchronously."""
    try:
        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        logger.error(f"Failed to save output to {output_filepath}: {e}")

def run_deepseek_ocr(image_paths, output_dir, tokenizer, model):
    """
    Runs DeepSeek-OCR inference on a list of image paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []
    
    # Use a thread pool for async file saving
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        
        for img_path in image_paths:
            try:
                image_output_dir = Path(output_dir) / Path(img_path).stem
                image_output_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Running DeepSeek-OCR on %s", img_path)
                
                # Inference (GPU bound, synchronous)
                with _patch_autocast():
                    generated_text = model.infer(
                        tokenizer=tokenizer,
                        prompt=DEFAULT_PROMPT,
                        image_file=str(img_path),
                        output_path=str(image_output_dir),
                        base_size=1024,
                        image_size=640,
                        crop_mode=True,
                        eval_mode=True,
                    )

                output_filepath = os.path.join(output_dir, os.path.basename(img_path) + ".txt")
                
                # Submit file saving to background thread
                futures.append(executor.submit(_save_ocr_result, output_filepath, generated_text))

                results.append(
                    {
                        "image_path": img_path,
                        "output_text_path": output_filepath,
                        "extracted_text_preview": generated_text[:200],
                        "status": "success",
                    }
                )
                logger.info(f"OCR successful for {img_path}. Output queued for saving to {output_filepath}")
            except Exception as e:
                results.append(
                    {
                        "image_path": img_path,
                        "status": "error",
                        "message": str(e),
                    }
                )
                logger.error(f"Error processing {img_path}: {e}", exc_info=True)
        
        # Ensure all writes finish before returning results
        concurrent.futures.wait(futures)

    return results


def process_pdf_for_ocr(pdf_path, image_output_dir, dpi=300):
    """
    Extracts images from each page of a PDF and saves them to a directory.
    Returns a list of paths to the saved images.
    """
    os.makedirs(image_output_dir, exist_ok=True)

    image_paths = []
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))

            img_filename = f"{Path(pdf_path).stem}_page_{page_num + 1}.png"
            img_path = os.path.join(image_output_dir, img_filename)
            pix.save(img_path)
            image_paths.append(img_path)
        doc.close()
        logger.info(f"Extracted {len(image_paths)} images from {pdf_path}")
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path} for image extraction: {e}", exc_info=True)
    return image_paths


def load_progress_log() -> Dict[str, Dict]:
    if not PROGRESS_LOG_PATH.exists():
        return {}
    try:
        with open(PROGRESS_LOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.warning(f"Progress log {PROGRESS_LOG_PATH} is corrupted. Starting fresh.")
        return {}


def save_progress_log(data: Dict[str, Dict]):
    PROGRESS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def mark_progress(progress: Dict[str, Dict], doc_key: str, status: str, **metadata):
    entry = {"status": status, "timestamp": datetime.utcnow().isoformat() + "Z"}
    entry.update(metadata)
    progress[doc_key] = entry
    save_progress_log(progress)


def collect_documents(input_path_obj: Path, recursive: bool = False) -> List[Path]:
    candidates: List[Path] = []
    if input_path_obj.is_dir():
        iterator = input_path_obj.rglob("*") if recursive else input_path_obj.iterdir()
        for item in iterator:
            if item.is_file() and item.suffix.lower() in (SUPPORTED_IMAGE_EXTS | {".pdf"}):
                candidates.append(item)
    elif input_path_obj.is_file() and input_path_obj.suffix.lower() in (SUPPORTED_IMAGE_EXTS | {".pdf"}):
        candidates.append(input_path_obj)
    else:
        logger.error(f"Unsupported input file type or path: {input_path_obj}")
    return sorted(candidates)


def cleanup_temp_dir(temp_dir: Optional[Path]):
    if temp_dir and temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Run DeepSeek-OCR on PDF or image files.")
    parser.add_argument("--input_path", required=True, help="Path to input PDF or image file(s). Can be a directory.")
    parser.add_argument(
        "--output_dir",
        default=str(ROOT_DIR / "data" / "output"),
        help="Directory to save OCR results.",
    )
    parser.add_argument(
        "--quantization_level",
        default=CONFIG["default_quantization"],
        help=f"Quantization level to use (e.g., 'fp16', 'Q5_K_M'). Default is '{CONFIG['default_quantization']}'.",
    )
    parser.add_argument("--device", default="cuda", help="Device to use for inference (e.g., 'cuda' or 'cpu').")
    parser.add_argument("--recursive", action="store_true", help="Recursively search for PDFs/images inside directories.")
    parser.add_argument("--reprocess", action="store_true", help="Process documents even if already completed.")
    parser.add_argument("--verbose_load", action="store_true", help="Enable verbose logging for model loading (HF info + telemetry disable).")

    args = parser.parse_args()

    if args.verbose_load:
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_info()
        logger.info("Verbose loading enabled. HF Telemetry disabled.")

    try:
        model_source, adapter_source, actual_quant_level = get_model_sources(args.quantization_level)
    except ValueError as e:
        logger.critical(f"Configuration error: {e}")
        return

    logger.info(f"Attempting to load model with quantization level: {actual_quant_level}")
    if adapter_source:
        logger.warning(
            "Adapter path configured (%s) but adapter loading is not yet supported; continuing without.",
            adapter_source,
        )

    try:
        logger.info("Initializing CUDA...")
        if torch.cuda.is_available():
            logger.info(f"CUDA Device Found: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("CUDA NOT DETECTED. Inference will be slow or fail.")

        logger.info("Loading Tokenizer...")
        tokenizer, model = _load_tokenizer_and_model(model_source, args.device, actual_quant_level)
        logger.info("Model & Tokenizer Loaded Successfully.")
    except RuntimeError as err:
        logger.critical("Unable to load DeepSeek-OCR model: %s", err)
        return

    temp_image_root = Path(args.output_dir).parent / "temp_pdf_images"
    temp_image_root.mkdir(parents=True, exist_ok=True)

    input_path_obj = Path(args.input_path)
    documents = collect_documents(input_path_obj, recursive=args.recursive)

    if not documents:
        logger.warning("No valid documents found for OCR.")
        return

    progress_log = load_progress_log()
    skip_completed = not args.reprocess

    overall_results = []

    for doc_path in documents:
        doc_key = str(doc_path.resolve())
        if skip_completed and progress_log.get(doc_key, {}).get("status") == "completed":
            logger.info(f"Skipping {doc_path} (already completed). Use --reprocess to override.")
            continue

        if doc_path.suffix.lower() == ".pdf" and doc_path.stat().st_size == 0:
            logger.error(f"Skipping empty PDF: {doc_path}")
            mark_progress(progress_log, doc_key, "error", message="Empty PDF file")
            continue

        doc_temp_dir = temp_image_root / doc_path.stem if doc_path.suffix.lower() == ".pdf" else None

        if doc_path.suffix.lower() == ".pdf":
            if doc_temp_dir:
                doc_temp_dir.mkdir(parents=True, exist_ok=True)
            image_paths = process_pdf_for_ocr(str(doc_path), str(doc_temp_dir))
        else:
            image_paths = [str(doc_path)]

        if not image_paths:
            logger.warning(f"No images extracted for {doc_path}.")
            mark_progress(progress_log, doc_key, "error", message="No images extracted")
            cleanup_temp_dir(doc_temp_dir)
            continue

        try:
            doc_results = run_deepseek_ocr(image_paths, args.output_dir, tokenizer, model)
            overall_results.extend(doc_results)
            mark_progress(progress_log, doc_key, "completed", pages=len(image_paths))
        except RuntimeError as e:
            logger.critical(f"DeepSeek-OCR aborted while processing {doc_path}: {e}")
            mark_progress(progress_log, doc_key, "error", message=str(e))
            cleanup_temp_dir(doc_temp_dir)
            return
        except Exception as e:
            logger.error(f"OCR failure for {doc_path}: {e}")
            mark_progress(progress_log, doc_key, "error", message=str(e))
        finally:
            cleanup_temp_dir(doc_temp_dir)

    summary_path = Path(args.output_dir) / "ocr_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(overall_results, f, indent=4)
    logger.info(f"OCR summary saved to {summary_path}")


if __name__ == "__main__":
    main()
