param(
    [string]$PythonExe = "C:\Dev\llm-research\deepseek-ocr\.venv\Scripts\python.exe",
    [string]$EvalDir = "C:\Dev\llm-research\deepseek-ocr\data\eval",
    [string]$OutputDir = "C:\Dev\llm-research\deepseek-ocr\data\benchmarks",
    [string]$QuantizationLevels = "",
    [string]$Device = "cuda"
)

Write-Host "`n=== DeepSeek-OCR Model Lifecycle :: Validate Quantization ===`n" -ForegroundColor Cyan

if (!(Test-Path $PythonExe)) { throw "Python executable not found at $PythonExe." }
if (!(Test-Path $EvalDir)) { throw "Eval directory missing at $EvalDir." }
if (!(Test-Path $OutputDir)) { New-Item -ItemType Directory -Path $OutputDir | Out-Null }

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outputPath = Join-Path $OutputDir ("benchmark_" + $timestamp + ".json")
$suitePath = "C:\Dev\llm-research\deepseek-ocr\scripts\benchmark_suite.py"

if (!(Test-Path $suitePath)) {
    throw "benchmark_suite.py not found at $suitePath."
}

$arguments = @(
    $suitePath,
    "--eval-dir", $EvalDir,
    "--output-path", $outputPath,
    "--device", $Device
)
if ($QuantizationLevels) {
    $arguments += @("--quantization-levels", $QuantizationLevels)
}

Write-Host "Executing benchmark suite..." -ForegroundColor Green
& $PythonExe @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Benchmark suite failed. Inspect console output for details."
}

Write-Host "`nBenchmark written to $outputPath`n" -ForegroundColor Cyan
