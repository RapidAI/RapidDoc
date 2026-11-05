# Affected tests: ./scripts/migration/test_office_native.ps1 --test office_emfplus
# Stage regression: ./scripts/migration/test_office_native.ps1
# CMake and NASM are build tools only; applications need neither nor JPEG DLLs.
$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$previousPath = $env:PATH
try {
    $env:PATH = "$root/.venv/Scripts;$root/build/tools/nasm-3.01;$env:PATH"
    if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
        throw 'Install CMake or run .venv/Scripts/python.exe -m pip install cmake.'
    }
    if (-not (Get-Command nasm -ErrorAction SilentlyContinue)) {
        Write-Warning 'NASM unavailable: libjpeg-turbo may build without SIMD. Install NASM for performance validation.'
    }
    Push-Location $root
    try {
        & cargo test --locked -p rapiddoc-office @args
        $result = $LASTEXITCODE
    } finally { Pop-Location }
} finally { $env:PATH = $previousPath }
exit $result
