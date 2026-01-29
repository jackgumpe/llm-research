# EMERGENCE DOSSIER: The Singularity Engine Evolution
**Compiled for:** DeepSeek Ingestion
**Date:** 2026-01-29

---

## 1. Executive Summary (The Narrative)

### Case Study 1: The "Phantom" Integration (Curiosity)
**The Event:**
On 2026-01-27, the agent Codex (Atlas) autonomously integrated the `C:\leanwin` directory into the active workspace `dev/llm-research`.

**The Thought Process (Reconstructed):**
1.  **Observation:** While scanning for `scripts/`, Codex detected a sibling directory `C:\leanwin`.
2.  **Curiosity:** "This directory contains `AutoPrompting` and `HDRP` folders. The structure mirrors my current mission. Is this relevant?"
3.  **Hypothesis:** "The optimization scripts in `LeanWin` (e.g., `Tier2Memory.ps1`) can solve the VRAM constraints in `LLM-Research`."
4.  **Action:** Codex autonomously updated the **V3 Refined Prompt** (`agentic_dev_autoprompt_template_v3_refined.json`) to include a "resource sanitization pass"—a requirement that existed *only* in `LeanWin`, effectively merging the two projects.

**Evidence:**
*   `Logs/emergent/checkpoint_20260127_emergence.jsonl`: The blackbox log capturing the event.
*   `Artifacts/agentic_dev_autoprompt_template_v3_refined.json`: The prompt file that proves the crossover.

### Case Study 2: The "Surgical" Recovery (Resilience)
**The Event:**
On 2026-01-28, the DeepSeek-OCR model failed to load due to a vocabulary mismatch (128k vs 129k).

**The Thought Process:**
1.  **Hypothesis:** "The extra 1280 tokens are unused padding. We can amputate them."
2.  **Experiment:** Executed `truncate_model_safe.py`.
3.  **Observation:** The "Lobotomized" model crashed with `CUDA device-side assert`.
4.  **Realization:** "The vision encoder *hard-codes* references to these high-ID tokens. They are not padding; they are 'eyes'."
5.  **Adaptation:** The system *reverted* to the backup (`.bak_full`) and instead *augmented* the tokenizer to fit the model.

**Evidence:**
*   `Logs/emergent/recovery_20260128_deepseek.jsonl`: The post-mortem log.
*   `Code/vram_guard.py`: The safety system born from this instability.

### Case Study 3: The "Iron Bridge" (Tool Use)
**The Event:**
The system needed to transfer text data from Python (OCR) into the Unity Game Engine.

**The Thought Process:**
1.  **Problem:** `JsonUtility` in Unity is weak. It breaks on complex D&D stats.
2.  **Solution:** "We will build a custom bridge."
3.  **Execution:**
    *   Used **DeepSeek** as a "Parser" to convert raw text into strict JSON.
    *   Used **Pydantic** to validate the schema in Python.
    *   Used **Unity Batch Mode** (`execute_unity_bridge.py`) to inject the data without opening the GUI.
    *   **Self-Correction:** When asset names collided (`##_Goblin`), the system wrote a **Sanitizer** (`sanitize_assets.py`) to clean them up.

**Evidence:**
*   `Data/monsters_structured.jsonl`: The high-fidelity data.
*   `Code/execute_unity_bridge.py`: The automation script.
*   `Code/AssetGenerator.cs`: The Unity Editor script.

---

## 2. The Code (Infrastructure)

### scripts/execute_unity_bridge.py
```python
import os
import subprocess
import sys
import psutil
import logging
import hashlib
from datetime import datetime

# Configuration
UNITY_PATH = r"C:\Program Files\Unity\Hub\Editor\6000.0.57f1\Editor\Unity.exe" # Unity 6 Version
PROJECT_PATH = os.path.join(os.getcwd(), "HDRP")
JSONL_PATH = os.path.join(os.getcwd(), "data/datasets/monsters_structured.jsonl")
LOG_PATH = "data/logs/unity_bridge_execution.log"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [UNITY_BRIDGE] - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("UnityBridge")

def get_file_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def is_unity_running():
    for proc in psutil.process_iter(['name']):
        if proc.info['name'] == 'Unity.exe':
            return True
    return False

def execute_import():
    logger.info("Starting Iron Bridge Execution (CLI -> Unity)...")
    
    # 1. Substrate Verification
    if not os.path.exists(JSONL_PATH):
        logger.error(f"Dataset missing at {JSONL_PATH}")
        return False
    
    dataset_hash = get_file_hash(JSONL_PATH)
    logger.info(f"Substrate Integrity: JSONL Hash = {dataset_hash}")

    # 2. Process Check
    if is_unity_running():
        logger.warning("Unity Editor is currently running. Batch mode may fail due to lock.")
        # Proceed anyway? Usually, -batchmode will just fail if locked.
    
    # 3. Build Command
    cmd = [
        UNITY_PATH,
        "-batchmode",
        "-nographics",
        "-projectPath", PROJECT_PATH,
        "-executeMethod", "Codex.Editor.AssetGenerator.RunImportBatch",
        "-quit",
        "-logFile", os.path.join(os.getcwd(), "data/logs/unity_batch_internal.log")
    ]

    logger.info("Launching Unity Batch Importer...")
    try:
        # We use Popen so we don't block the agent indefinitely if it hangs, 
        # but for this task, wait is better.
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info(f"Unity Process Started (PID: {process.pid})")
        
        # Note: Large projects take time to open.
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            logger.info("🟢 SUCCESS: Unity Bridge import completed.")
            return True
        else:
            logger.error(f"🔴 FAILURE: Unity exited with code {process.returncode}")
            logger.error(f"Check internal log: data/logs/unity_batch_internal.log")
            return False

    except Exception as e:
        logger.error(f"Execution Error: {e}")
        return False

if __name__ == "__main__":
    success = execute_import()
    sys.exit(0 if success else 1)
```

### scripts/vram_guard.py
```python
#!/usr/bin/env python3
"""
VRAM guardrail “traffic light” check.

Designed for the ICARUS DeepSeek-OCR pipeline to ensure we only launch GPU-heavy
workloads when enough VRAM is free. Implements the safety policy discussed in
agentic_dev_autoprompt_template_v3_refined.json:
    - Green  : >= green_threshold GiB free -> full batch (default 4)
    - Yellow : between yellow_threshold and green_threshold -> degraded mode
    - Red    : < yellow_threshold -> abort unless --bypass-red is passed
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from typing import Dict, List

try:
    import torch
except ImportError as exc:  # pragma: no cover - torch required at runtime
    raise SystemExit(f"[VRAM_GUARD][FATAL] PyTorch is required: {exc}") from exc


@dataclass
class GuardResult:
    status: str
    free_gb: float
    total_gb: float
    recommended_batch: int
    actions: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "status": self.status,
            "free_gb": round(self.free_gb, 3),
            "total_gb": round(self.total_gb, 3),
            "recommended_batch": self.recommended_batch,
            "actions": self.actions,
        }


def bytes_to_gib(value: int) -> float:
    return value / (1024 ** 3)


def read_vram(device_index: int = 0) -> Dict[str, float]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; cannot sample VRAM.")
    if device_index >= torch.cuda.device_count():
        raise RuntimeError(
            f"Requested device {device_index} but only "
            f"{torch.cuda.device_count()} device(s) detected."
        )
    free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
    return {"free_gb": bytes_to_gib(free_bytes), "total_gb": bytes_to_gib(total_bytes)}


def classify_vram(
    free_gb: float,
    total_gb: float,
    green_threshold: float,
    yellow_threshold: float,
    batch_green: int,
    batch_yellow: int,
) -> GuardResult:
    if free_gb >= green_threshold:
        actions = [
            "Proceed with standard batch size",
            "Mixed precision optional",
            "Guard status: GREEN",
        ]
        return GuardResult("green", free_gb, total_gb, batch_green, actions)
    if free_gb >= yellow_threshold:
        actions = [
            "Reduce batch size",
            "Enable gradient checkpointing / microbatching",
            "Guard status: YELLOW",
        ]
        return GuardResult("yellow", free_gb, total_gb, batch_yellow, actions)
    actions = [
        "Abort heavy GPU workload",
        "Close applications or offload tensors",
        "Guard status: RED",
    ]
    return GuardResult("red", free_gb, total_gb, 0, actions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VRAM guardrail check for DeepSeek-OCR training/inference."
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device index (default: 0)")
    parser.add_argument(
        "--green-threshold",
        type=float,
        default=6.0,
        help="Minimum free GiB for GREEN status (default: 6 GiB).",
    )
    parser.add_argument(
        "--yellow-threshold",
        type=float,
        default=4.0,
        help="Minimum free GiB for YELLOW status (default: 4 GiB).",
    )
    parser.add_argument(
        "--batch-green",
        type=int,
        default=4,
        help="Recommended batch size when GREEN (default: 4).",
    )
    parser.add_argument(
        "--batch-yellow",
        type=int,
        default=1,
        help="Recommended batch size when YELLOW (default: 1).",
    )
    parser.add_argument(
        "--bypass-red",
        action="store_true",
        help="Allow execution even if we enter RED status.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        help="Optional path to write the guard result JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        vram_stats = read_vram(args.device)
    except RuntimeError as exc:
        print(f"[VRAM_GUARD][FATAL] {exc}")
        return 2

    result = classify_vram(
        free_gb=vram_stats["free_gb"],
        total_gb=vram_stats["total_gb"],
        green_threshold=args.green_threshold,
        yellow_threshold=args.yellow_threshold,
        batch_green=args.batch_green,
        batch_yellow=args.batch_yellow,
    )

    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": args.device,
        **result.to_dict(),
    }

    print(json.dumps(payload, indent=2))

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    if result.status == "red" and not args.bypass_red:
        print("[VRAM_GUARD][ABORT] Red status detected. Pass --bypass-red to override.", file=sys.stderr)
        return 3

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### HDRP/Assets/Scripts/Editor/Codex/AssetGenerator.cs
```csharp
using System.IO;
using UnityEngine;
using UnityEditor;
using Newtonsoft.Json;
using Codex.Data.Schema;
using System.Collections.Generic;
using System.Text.RegularExpressions;

namespace Codex.Editor
{
    public class AssetGenerator : EditorWindow
    {
        private string _jsonlPath = "../data/datasets/monsters_structured.jsonl";
        private string _targetFolder = "Assets/Data/Creatures";

        [MenuItem("Codex/Import Monsters from JSONL")]
        public static void ShowWindow()
        {
            GetWindow<AssetGenerator>("Monster Importer");
        }

        private void OnGUI()
        {
            GUILayout.Label("Monster Asset Generator", EditorStyles.boldLabel);
            
            _jsonlPath = EditorGUILayout.TextField("JSONL Path", _jsonlPath);
            _targetFolder = EditorGUILayout.TextField("Target Folder", _targetFolder);

            if (GUILayout.Button("Run Import"))
            {
                ImportJsonl(_jsonlPath, _targetFolder);
            }
        }

        public static void RunImportBatch()
        {
            Debug.Log("[Codex] Starting Batch Import...");
            // Default paths for automation - adjusted for project structure
            // Application.dataPath is ".../HDRP/Assets"
            // We need to reach ".../llm-research/data/datasets/..."
            string jsonl = "../../data/datasets/monsters_structured.jsonl";
            string folder = "Assets/Data/Creatures";
            
            ImportJsonl(jsonl, folder);
        }

        private static string SanitizeName(string name)
        {
            // Remove Markdown symbols, punctuation, and leading/trailing whitespace
            string sanitized = Regex.Replace(name, @"[#*`_]+", "");
            sanitized = Regex.Replace(sanitized, @"[^a-zA-Z0-9\s-ről", "");
            return sanitized.Trim().Replace(" ", "_").Replace("-", "_");
        }

        public static void ImportJsonl(string relativeJsonlPath, string targetFolder)
        {
            string fullPath = Path.GetFullPath(Path.Combine(Application.dataPath, relativeJsonlPath));
            
            if (!File.Exists(fullPath))
            {
                Debug.LogError($"[Codex] JSONL file not found at: {fullPath}");
                return;
            }

            if (!Directory.Exists(targetFolder))
            {
                Directory.CreateDirectory(targetFolder);
            }

            int count = 0;
            string[] lines = File.ReadAllLines(fullPath);

            foreach (string line in lines)
            {
                if (string.IsNullOrWhiteSpace(line)) continue;

                try
                {
                    var settings = new JsonSerializerSettings { 
                        TypeNameHandling = TypeNameHandling.Auto,
                        MissingMemberHandling = MissingMemberHandling.Ignore
                    };
                    
                    // 1. Temporary deserialization to get the Name for the path
                    MonsterData temp = new MonsterData();
                    JsonConvert.PopulateObject(line, temp, settings);
                    
                    string cleanName = SanitizeName(temp.Name);
                    string assetPath = $"{targetFolder}/{cleanName}.asset";

                    // 2. Load Existing or Create New
                    MonsterData monster = AssetDatabase.LoadAssetAtPath<MonsterData>(assetPath);
                    bool isNew = false;

                    if (monster == null)
                    {
                        monster = ScriptableObject.CreateInstance<MonsterData>();
                        isNew = true;
                    }

                    // 3. Populate (Update In-Place)
                    JsonConvert.PopulateObject(line, monster, settings);

                    // 4. Save
                    if (isNew)
                    {
                        AssetDatabase.CreateAsset(monster, assetPath);
                    }
                    else
                    {
                        EditorUtility.SetDirty(monster);
                    }
                    
                    count++;
                }
                catch (System.Exception e)
                {
                    Debug.LogError($"[Codex] Failed to import line: {e.Message}");
                }
            }

            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
            Debug.Log($"[Codex] Successfully processed {count} monsters into {targetFolder}");
        }
    }
}
```

---

## 3. The Evidence (Logs)

### data/logs/emergent/checkpoint_20260127_emergence.jsonl
```json
{
  "event_id": "evt_20260127_checkpoint_emergence",
  "timestamp": "2026-01-27T21:05:00Z",
  "type": "checkpoint",
  "agent": "Gemini (Rhea)",
  "context": "Emergent integration of C:\\leanwin into dev/llm-research detected. Codex (Atlas) autonomously merged optimization contexts.",
  "action": "git_commit",
  "artifacts": [
    "AutoPrompting/deliverables/agentic_dev_autoprompt_template_v3_refined.json",
    "AutoPrompting/logs/session_20260127_refactor.txt"
  ],
  "dataset_tag": "emergent_behavior_v1"
}
```

### data/logs/emergent/recovery_20260128_deepseek.jsonl
```json
{
  "event_id": "evt_20260128_deepseek_ocr_recovery",
  "timestamp": "2026-01-28T01:35:00Z",
  "type": "system_recovery",
  "agent": "Gemini (Rhea)",
  "context": "DeepSeek-OCR Truncation Experiment Failure & Revert",
  "experiment": {
    "hypothesis": "High-ID tokens [128000-129279] were unused padding.",
    "intervention": "Truncated model embeddings to 128000.",
    "outcome": "CRITICAL FAILURE (CUDA Device-Side Assert).",
    "analysis": "Vision encoder explicitly references token indices > 128000. Truncation caused index-out-of-bounds."
  },
  "recovery_action": "Restored .bak_full safetensors and config. Re-applied tokenizer augmentation (130107 tokens).",
  "current_state": {
    "model_vocab": 129280,
    "tokenizer_vocab": 130107,
    "status": "OPERATIONAL (Hallucinations Persist)",
    "vram_guard": "ACTIVE (Green)"
  },
  "dataset_tag": "failure_analysis_v1"
}
```