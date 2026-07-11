param(
    [string]$Data = "",
    [ValidateSet("instant", "full")]
    [string]$Mode = "instant"
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
        Write-Host ""
        Write-Host "Có nhiều file parquet. Chạy rõ path:" -ForegroundColor Yellow
        Write-Host '.\run.ps1 -Data "C:\duong-dan\btc_15m.parquet"'
        $files | ForEach-Object { Write-Host " - $($_.FullName)" }
        exit 2
    }
    else {
        Write-Host ""
        Write-Host "Không thấy file parquet." -ForegroundColor Yellow
        Write-Host "Copy BTC 15m parquet vào folder này, hoặc chạy:"
        Write-Host '.\run.ps1 -Data "C:\duong-dan\btc_15m.parquet"'
        exit 2
    }
}

$python = $null
if (Get-Command py -ErrorAction SilentlyContinue) {
    $python = "py"
}
elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $python = "python"
}
else {
    throw "Không tìm thấy Python. Cần Python 3.11–3.14."
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Creating .venv..."
    if ($python -eq "py") {
        & py -3.12 -m venv .venv
        if ($LASTEXITCODE -ne 0) {
            & py -3 -m venv .venv
        }
    }
    else {
        & python -m venv .venv
    }
}

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$marker = Join-Path $PSScriptRoot ".venv\.vectorbt_ready"

if (-not (Test-Path $marker)) {
    Write-Host "Installing vectorbt + parquet support..."
    & $venvPython -m pip install --upgrade pip
    & $venvPython -m pip install --upgrade "vectorbt[rust]==1.1.0" pyarrow
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Rust wheel không cài được; fallback Numba." -ForegroundColor Yellow
        & $venvPython -m pip install --upgrade "vectorbt==1.1.0" pyarrow
        if ($LASTEXITCODE -ne 0) {
            throw "Cài dependency thất bại."
        }
    }
    New-Item -ItemType File -Path $marker -Force | Out-Null
}

Write-Host ""
Write-Host "RUNNING FAST SEARCH" -ForegroundColor Cyan
Write-Host "Data: $Data"
Write-Host "Mode: $Mode"
Write-Host ""

& $venvPython ".\search.py" `
    --data $Data `
    --mode $Mode `
    --output ".\output"

if ($LASTEXITCODE -ne 0) {
    throw "Search thất bại. Xem ERROR phía trên."
}

Write-Host ""
Write-Host "DONE: .\output\FINAL_RESULT.txt" -ForegroundColor Green
