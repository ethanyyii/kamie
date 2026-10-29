# Starts the production TTS adapter (OmniVoice) on port 5050.
#
# The Python environment still lives under tts-test\omnivoice\.venv -- it was
# built there in v8 and moving a venv is a needless risk, so this script just
# points at it. Nothing else in the tree is touched.
#
#   .\start_tts.ps1              # num_step 16 (default; the user's chosen setting)
#   .\start_tts.ps1 -NumStep 32  # the slower original, kept switchable

param(
    # 16, not 32: synthesis time is linear in steps, so this halves TTS from
    # ~1.06s to ~0.53s, and the user judged it acceptable on a real listen.
    [int]$NumStep = 16,
    [int]$Port = 5050,
    # AIRI's speech Base URL was moved to 5050 in v12, so the alias is no longer
    # needed and defaults to off. Pass -AliasPorts "5051" to bring it back if a
    # stale setting ever points there again.
    [string]$AliasPorts = "",
    [switch]$Normalize
)

$venv = if ($env:PYTHON_EXE) { $env:PYTHON_EXE } else { "python" }
$adapter = Join-Path $PSScriptRoot "openai_tts_adapter.py"

$env:OMNIVOICE_NUM_STEP = $NumStep
$env:OMNIVOICE_PORT = $Port
$env:OMNIVOICE_ALIAS_PORTS = $AliasPorts
if ($Normalize) { $env:OMNIVOICE_NORMALIZE = "1" } else { $env:OMNIVOICE_NORMALIZE = "0" }

Write-Host "OmniVoice TTS -> http://127.0.0.1:$Port (alias: $AliasPorts)  (num_step=$NumStep, normalize=$($env:OMNIVOICE_NORMALIZE))"
& $venv $adapter
