$ErrorActionPreference = 'Stop'
$stageRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$python313 = Join-Path $env:LOCALAPPDATA 'Programs/Python/Python313/python.exe'
if (-not (Test-Path -LiteralPath $python313)) {
    throw 'Python 3.13 executable not found. Set python313 to the installed interpreter.'
}
$env:PYTHONPATH = (Join-Path $stageRoot '.cuda-runtime') + ';' + (Join-Path $stageRoot '.runtime') + ';' + $stageRoot
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$env:PYTHONUNBUFFERED = '1'
& $python313 -c "import torch; assert torch.cuda.is_available(), 'CUDA runtime is unavailable'; print(torch.__version__, torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { throw 'CUDA preflight failed' }
& $python313 (Join-Path $PSScriptRoot 'run_benchmarks.py')
if ($LASTEXITCODE -ne 0) { throw 'Experiment failed; do not publish partial results as final' }
& $python313 -m dl.training.finalize_evidence
if ($LASTEXITCODE -ne 0) { throw 'Evidence finalization failed' }
