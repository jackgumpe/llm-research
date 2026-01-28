# Automated Debugging Framework (ADF)
**Version:** 1.0.0
**Date:** 2026-01-28
**Authors:** Rhea (Gemini), Atlas (Claude), Omen (DeepSeek)

## Executive Summary
The **Automated Debugging Framework (ADF)** is a suite of Python scripts designed to enforce **Observability & Control Theory** principles within the `llm-research` project (Singularity Engine). It shifts the operational paradigm from "Reactive Debugging" to "Proactive Guardrailing," ensuring that high-value GPU workloads (like VLM training) only execute when environmental conditions are verified safe.

## Core Directives
1.  **If you can't measure it, you can't manage it.**
2.  **System Homeostasis** is prioritized over raw throughput.
3.  **Fail Fast, Fail Loud, Fail Safe.**

---

## Component Architecture

### 1. Resource Sentinel (`scripts/resource_sanitizer.py`)
*   **Role:** The "Air Traffic Controller" / Pre-flight Check.
*   **Function:** Scans system RAM, CPU, and running processes *before* any heavy workload begins.
*   **Capabilities:**
    *   Flags "Heavy" processes (>1GB RAM).
    *   Calculates `available_memory_gb` (Critical for VLM loading).
    *   (Planned) Can proactively terminate non-essential apps (e.g., browsers) if RAM < 4GB.
*   **Artifact:** `data/logs/resource_sanitization_report.json`

### 2. VRAM Guard (`scripts/vram_guard.py`)
*   **Role:** The "Traffic Light" Safety Switch.
*   **Function:** Queries NVIDIA driver (via `torch.cuda`) for real-time VRAM availability.
*   **Policy (16GB Laptop Profile):**
    *   🟢 **GREEN (> 6GB):** Proceed with standard Batch Size (4).
    *   🟡 **YELLOW (4-6GB):** Proceed with degraded parameters (Batch=1, Gradient Checkpointing).
    *   🔴 **RED (< 4GB):** **ABORT IMMEDIATE**. Prevent system freeze.
*   **Usage:** Can be imported (`from vram_guard import classify_vram`) or run as a standalone CLI check (`python scripts/vram_guard.py`).

### 3. Log Scanner (`scripts/log_scanner.py`)
*   **Role:** The "Black Box" Analyst.
*   **Function:** Post-mortem (or pre-flight) analysis of logs to reconstruct timelines and attribute blame.
*   **Capabilities:**
    *   **Timeline Reconstruction:** Merges timestamps from multiple log files.
    *   **Crash Attribution:** Regex heuristics to classify failures (VLM vs ImageGen vs Unity).
    *   **Evidence Citation:** Maps every claim to a specific `file:line`.
*   **Artifact:** `data/logs/incident_timeline.json`

---

## Integration Patterns

### A. The "Safe Launch" Pattern
All critical entry points (e.g., `icarus_revelation.py`, `deepseek_direct.py`) MUST implement this flow:

```python
# 1. Pre-flight Resource Check
import scripts.resource_sanitizer as rs
report = rs.scan_resources()
if report['memory_available_gb'] < 4.0:
    print("CRITICAL: System RAM too low.")
    # Optional: rs.sanitize()

# 2. VRAM Guardrail
from scripts.vram_guard import VRAMGuard
guard = VRAMGuard()
status, free_mb = guard.check_vram()
if status == "RED":
    sys.exit("VRAM GUARD: ABORTING LAUNCH")
elif status == "YELLOW":
    CONFIG.batch_size = 1 # Adaptive Degrade

# 3. Execution
model = load_model(...)
```

### B. The "Crash Loop" Pattern
If a run fails (exit code != 0):
1.  Trigger `scripts/log_scanner.py`.
2.  Read `data/logs/incident_timeline.json`.
3.  If `decision == "VLM_TRAINING"` and `first_fatal` contains "OOM":
    *   Auto-tune: Reduce `batch_green` in `vram_guard.py` for next run.

---

## Case Study: DeepSeek-OCR Recovery
*   **Incident:** Tensor Shape Mismatch (128000 vs 129280).
*   **Hypothesis:** Extra 1280 tokens were unused padding.
*   **Intervention (Failed):** Truncated model to 128000.
*   **Result:** `CUDA device-side assert`. Vision encoder output mapped to the truncated range.
*   **Recovery:** Restored model, augmented tokenizer to 130107.
*   **Lesson:** **Do not amputate "Ghost" vectors without forensic proof of inactivity.**
*   **Artifact:** `data/logs/emergent/recovery_20260128_deepseek.jsonl`

---

## Future Roadmap
1.  **Token Forensics:** Deep dive into the 1280 "Ghost" tokens to map them to visual concepts (reduce hallucinations).
2.  **Unity Data Bridge:** Refactor `HDRP/Assets/Scripts` to use `Newtonsoft.Json`, enabling robust serialization of the data we ingest via this pipeline.
