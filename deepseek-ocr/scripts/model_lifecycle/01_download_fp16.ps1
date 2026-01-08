param(
    [string]$PythonExe = "C:\Dev\llm-research\deepseek-ocr\.venv\Scripts\python.exe",
    [string]$ConfigPath = "C:\Dev\llm-research\deepseek-ocr\models\config.json"
)

Write-Host "`n=== DeepSeek-OCR Model Lifecycle :: Download FP16 Assets ===`n" -ForegroundColor Cyan

if (!(Test-Path $PythonExe)) {
    throw "Python executable not found at $PythonExe. Activate the virtual environment or adjust the -PythonExe parameter."
}
if (!(Test-Path $ConfigPath)) {
    throw "Config file not found at $ConfigPath."
}

$config = Get-Content $ConfigPath | ConvertFrom-Json
$fp16Config = $config.models.fp16
if (-not $fp16Config) {
    throw "fp16 entry missing from $ConfigPath."
}

function Invoke-HuggingFaceDownload {
    param(
        [string]$RepoId,
        [string]$TargetDir
    )
    Write-Host "Downloading $RepoId -> $TargetDir" -ForegroundColor Green
    if (!(Test-Path $TargetDir)) {
        New-Item -ItemType Directory -Path $TargetDir | Out-Null
    }

    $downloadScript = @"
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id=r"$RepoId",
    local_dir=r"$TargetDir",
    allow_patterns=["*.safetensors", "*.json", "*.py"],
    tqdm_class=None
)
"@

    & $PythonExe -c $downloadScript
    if ($LASTEXITCODE -ne 0) {
        throw "huggingface_hub snapshot_download failed for $RepoId."
    }
}

Invoke-HuggingFaceDownload -RepoId $fp16Config.huggingface_repo_id -TargetDir $fp16Config.model_path

if ($fp16Config.adapter_repo_id) {
    Invoke-HuggingFaceDownload -RepoId $fp16Config.adapter_repo_id -TargetDir $fp16Config.adapter_path
}

Write-Host "`nFP16 assets synchronized successfully.`n" -ForegroundColor Cyan
