param(
    [string]$ConfigPath = "C:\Dev\llm-research\deepseek-ocr\models\config.json",
    [string]$QuantizeExe = "C:\tools\llama.cpp\build\bin\quantize.exe",
    [string]$SourceQuantLevel = "fp16",
    [string]$TargetQuantLevel = "Q5_K_M"
)

Write-Host "`n=== DeepSeek-OCR Model Lifecycle :: Quantize GGUF ===`n" -ForegroundColor Cyan

if (!(Test-Path $ConfigPath)) { throw "Missing config file at $ConfigPath." }
if (!(Test-Path $QuantizeExe)) {
    throw "quantize.exe not found at $QuantizeExe. Build llama.cpp and update the -QuantizeExe parameter."
}

$config = Get-Content $ConfigPath | ConvertFrom-Json
$sourceConfig = $config.models.$SourceQuantLevel
$targetConfig = $config.models.$TargetQuantLevel

if (-not $sourceConfig) { throw "Source level '$SourceQuantLevel' missing from config." }
if (-not $targetConfig) { throw "Target level '$TargetQuantLevel' missing from config." }

$inputModel = Join-Path $sourceConfig.model_path "ggml-model-f16.gguf"
if (!(Test-Path $inputModel)) {
    Write-Warning "Default f16 GGUF not found at $inputModel. Attempting to locate *.safetensors instead."
    $safetensors = Get-ChildItem $sourceConfig.model_path -Filter *.safetensors -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -eq $safetensors) {
        throw "Could not locate a model file inside $($sourceConfig.model_path)."
    }
    $inputModel = $safetensors.FullName
}

if (!(Test-Path $targetConfig.model_path)) {
    New-Item -ItemType Directory -Path $targetConfig.model_path | Out-Null
}

$outputModel = Join-Path $targetConfig.model_path ("deepseek-ocr-" + $TargetQuantLevel + ".gguf")
$arguments = @(
    "`"$inputModel`"",
    "`"$outputModel`"",
    $TargetQuantLevel
)

Write-Host "Running $QuantizeExe $($arguments -join ' ')" -ForegroundColor Green
& $QuantizeExe @arguments
if ($LASTEXITCODE -ne 0) {
    throw "llama.cpp quantize command failed."
}

Write-Host "Quantized model created at $outputModel" -ForegroundColor Cyan
