import psutil
import json
import os
import time
from datetime import datetime

# Configuration
LOG_FILE = "data/logs/resource_sanitization_report.json"
MEMORY_THRESHOLD_MB = 1024  # Flag apps using > 1GB RAM
TARGET_PROCESSES = ["chrome.exe", "msedge.exe", "firefox.exe", "unity.exe", "python.exe"]

def scan_resources():
    report = {
        "timestamp": datetime.now().isoformat(),
        "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "memory_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
        "cpu_percent": psutil.cpu_percent(interval=1),
        "flagged_processes": []
    }

    print(f"[*] Scanning Resources... (Available RAM: {report['memory_available_gb']} GB)")

    for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
        try:
            mem_mb = proc.info['memory_info'].rss / (1024 * 1024)
            if mem_mb > MEMORY_THRESHOLD_MB or proc.info['name'].lower() in TARGET_PROCESSES:
                report["flagged_processes"].append({
                    "pid": proc.info['pid'],
                    "name": proc.info['name'],
                    "memory_mb": round(mem_mb, 2)
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    # Sort by memory usage
    report["flagged_processes"].sort(key=lambda x: x["memory_mb"], reverse=True)
    return report

def save_report(report):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[+] Report saved to {LOG_FILE}")

if __name__ == "__main__":
    report = scan_resources()
    save_report(report)
