param(
    [string]$Data = "",
    [ValidateSet("coarse", "full")]
    [string]$Mode = "coarse",
    [switch]$Resume,
    [int]$BatchSize = 48,
    [int]$MaxConfigs = 0,
    [double]$Fee = 0.0005,
    [double]$Slippage = 0.0002,
    [int]$MinTradesRanking = 100
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if ([string]::IsNullOrWhiteSpace($Data)) {
    $files = Get-ChildItem -Path $PSScriptRoot -Filter *.parquet -File -Recurse |
        Where-Object { $_.FullName -notmatch "\\output\\" }

    if ($files.Count -eq 1) {
        $Data = $files[0].FullName
    }
    elseif ($files.Count -gt 1) {
        Write-Host "Có nhiều file parquet. Chỉ rõ file bằng -Data." -ForegroundColor Yellow
        $files | ForEach-Object { Write-Host " - $($_.FullName)" }
        exit 2
    }
    else {
        Write-Host "Không thấy file parquet." -ForegroundColor Yellow
        Write-Host 'Copy data vào folder hoặc chạy:'
        Write-Host '.\run.ps1 -Data "C:\duong-dan\btc_15m.parquet"'
        exit 2
    }
}

$launcher = $null
if (Get-Command py -ErrorAction SilentlyContinue) {
    $launcher = "py"
}
elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $launcher = "python"
}
else {
    throw "Không tìm thấy Python 3.11–3.14."
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Creating .venv..."
    if ($launcher -eq "py") {
        & py -3.12 -m venv .venv
        if ($LASTEXITCODE -ne 0) {
            & py -3 -m venv .venv
        }
    }
    else {
        & python -m venv .venv
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Không tạo được virtual environment."
    }
}

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$marker = Join-Path $PSScriptRoot ".venv\.ready-v1"

if (-not (Test-Path $marker)) {
    Write-Host "Installing dependencies..."
    & $python -m pip install --upgrade pip
    & $python -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Rust wheel không tương thích. Fallback vectorbt Numba..." -ForegroundColor Yellow
        & $python -m pip install --upgrade "vectorbt==1.1.0" pyarrow pytest
        if ($LASTEXITCODE -ne 0) {
            throw "Cài dependency thất bại."
        }
    }
    New-Item -ItemType File -Path $marker -Force | Out-Null
}

$argsList = @(
    "-m", "btsearch.cli",
    "--data", $Data,
    "--mode", $Mode,
    "--batch-size", $BatchSize,
    "--fee", $Fee,
    "--slippage", $Slippage,
    "--min-trades-ranking", $MinTradesRanking,
    "--output", ".\output"
)

if ($Resume) {
    $argsList += "--resume"
}
if ($MaxConfigs -gt 0) {
    $argsList += @("--max-configs", $MaxConfigs)
}

Write-Host ""
Write-Host "VECTORBT STRATEGY RESEARCH" -ForegroundColor Cyan
Write-Host "Data: $Data"
Write-Host "Mode: $Mode"
Write-Host "Resume: $Resume"
Write-Host ""

& $python @argsList
if ($LASTEXITCODE -ne 0) {
    throw "Research run thất bại. Xem lỗi phía trên."
}

Write-Host ""
Write-Host "DONE: .\output\FINAL_REPORT.txt" -ForegroundColor Green
