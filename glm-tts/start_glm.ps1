# Starts the GLM-TTS adapter on port 5050 -- the v50 production candidate.
#
# Same HTTP contract as omnivoice-tts\start_tts.ps1, so AIRI needs no settings
# change: run ONE of the two scripts, never both on the same port. The OmniVoice
# path is untouched and remains the fallback engine (ENVIRONMENT.md section 4).
#
#   .\start_glm.ps1                       # young2 voice (the v49 decision), seed 0
#   .\start_glm.ps1 -Prompt gprobe        # the fallback voice the client named
#   .\start_glm.ps1 -Seed 3               # another sampling seed
#   .\start_glm.ps1 -KeepSpeechTokenizer  # do not drop the 1.6 GB speech tokenizer after prompt encoding
#   .\start_glm.ps1 -NoTrimHead           # leave GLM's leading silence in
#   .\start_glm.ps1 -JumpReroll           # regenerate a suspected voice jump before answering
#   .\start_glm.ps1 -PunctDowngrade       # soften mid-segment 。！？ to ， (shorter pauses)
#
# Everything here is also switchable at runtime through POST /config.

param(
    [int]$Port = 5050,
    [string]$Prompt = "young2",
    [int]$Seed = 0,
    [string]$AliasPorts = "",
    [switch]$KeepSpeechTokenizer,
    [switch]$NoTrimHead,
    [switch]$JumpReroll,
    [switch]$PunctDowngrade,
    [ValidateSet('fp16', 'fp32')]
    [string]$LlmDtype = 'fp16',
    # v83 (RTX 3090 24 GB): per-process VRAM cap as a fraction of the card. ON by default
    # (0.35 = 8.6 GB; GLM needs ~4.3 GB steady, 6.4 GB peak while loading). Without it a
    # one-character segment (「嗯？」) made the flow step reserve ~37 GB, Windows backed it
    # with system RAM, 40 s per segment or a cuDNN failure, and GLM stayed dead until
    # restart = she replies with no voice (mainline-v83\incident-gpu-reset-0927-v83.md).
    # The 12 GB card could never grant that allocation, which is why it never showed up.
    # Resource only. Pass -VramFraction 0 to switch the cap off (the pre-v83 behaviour).
    [double]$VramFraction = 0.35
)

$python = if ($env:PYTHON_EXE) { $env:PYTHON_EXE } else { "python" }
$adapter = Join-Path $PSScriptRoot "glm_tts_adapter.py"

$env:GLM_PORT = $Port
$env:GLM_PROMPT = $Prompt
$env:GLM_SEED = $Seed
$env:GLM_ALIAS_PORTS = $AliasPorts
$env:GLM_LLM_DTYPE = $LlmDtype
$env:GLM_UNLOAD_SPEECH_TOKENIZER = if ($KeepSpeechTokenizer) { "0" } else { "1" }
$env:GLM_TRIM_HEAD = if ($NoTrimHead) { "0" } else { "1" }
$env:GLM_JUMP_REROLL = if ($JumpReroll) { "1" } else { "0" }
$env:GLM_PUNCT_DOWNGRADE = if ($PunctDowngrade) { "1" } else { "0" }
$env:GLM_VRAM_FRACTION = if ($VramFraction -gt 0) { $VramFraction.ToString([Globalization.CultureInfo]::InvariantCulture) } else { "" }
$env:PYTHONIOENCODING = "utf-8"

Write-Host "GLM-TTS -> http://127.0.0.1:$Port (alias: $AliasPorts)  (prompt=$Prompt, seed=$Seed, llm=$LlmDtype, unload_speech_tokenizer=$($env:GLM_UNLOAD_SPEECH_TOKENIZER), trim_head=$($env:GLM_TRIM_HEAD), jump_reroll=$($env:GLM_JUMP_REROLL))"
& $python $adapter
