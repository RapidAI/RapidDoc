# Build the native PDF/Office preview and retain canonical Office notices in wheel.
param([string]$OutputDirectory='build/office-docx-wheels', [switch]$MigrationTools)
$ErrorActionPreference='Stop'
$root=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$destination=Join-Path $root 'bindings/python/python/rapid_doc_rust/licenses/office'
New-Item -ItemType Directory -Path $destination -Force | Out-Null
Get-ChildItem -LiteralPath (Join-Path $root 'rust/rapiddoc-office/licenses') | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination $destination -Recurse -Force
}
foreach ($name in @('office-native-provenance.json','office-image-dependency-notices.json','office-chart-dependency-notices.json')) {
    Copy-Item -LiteralPath (Join-Path $root "packaging/$name") -Destination $destination -Force
}
& (Join-Path $root '.venv/Scripts/python.exe') (Join-Path $root 'scripts/migration/compact_release_notices.py')
if ($LASTEXITCODE -ne 0) { throw 'Third-party notice consolidation failed' }
$oldPath=$env:PATH
try {
    $env:PATH="$root/.venv/Scripts;$root/build/tools/nasm-3.01;$env:PATH"
    Push-Location $root
    try {
        $buildArguments=@('-m','maturin','build','--manifest-path','bindings/python/Cargo.toml','--release','--locked','--out',$OutputDirectory,'-i','.venv/Scripts/python.exe')
        if ($MigrationTools) { $buildArguments+=@('--features','migration-tools') }
        & .venv/Scripts/python.exe @buildArguments
        $buildExit=$LASTEXITCODE
    } finally { Pop-Location }
} finally { $env:PATH=$oldPath }
exit $buildExit
