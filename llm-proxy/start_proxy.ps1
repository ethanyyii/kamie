# Starts the LLM reasoning proxy on port 11435, in front of Ollama.
#
# AIRI's LLM Base URL points here instead of at 11434; the proxy adds the one
# field Ollama's OpenAI-compatible endpoint actually honours for turning qwen3's
# reasoning off. Reuses the omnivoice venv rather than building another one --
# it already has fastapi, uvicorn and httpx.
#
#   .\start_proxy.ps1              # thinking off (fast, default)
#   .\start_proxy.ps1 -Thinking on # thinking on (model default)
#   .\start_proxy.ps1 -ModelOverride ""   # v30: do NOT rewrite the model (use AIRI's own setting)
#
# v30 W1: the default base is qwen3.5:4b. AIRI's settings page still binds
# qwen3:1.7b (automation must not touch the settings page), so the proxy
# rewrites the model field on every request unless -ModelOverride "" is given.

#
# v51 (planner ruling 2026-09-08): the user-identity seam is ON by default = "P3".
#   -UserPersona     one sentence appended to the system prompt saying who the user is
#   -UserNamePrefix  put in front of every real user message (after AIRI's timestamp)
# Why both (measured on the real desktop path, 20-turn scripts): third-person
# misuse of User fell from 51/80 turns to 3/60 only with both on; the sentence
# alone left the reading unchanged (15/60 still "User會罵你"), the prefix alone made
# her stop saying the name (1 correct in 60 = hiding, not fixing). Costs recorded in
# ENVIRONMENT §6/§10: the prefix was once read as the user calling HER User
# ("你叫我User?"), SillyTavern #557 is the upstream precedent, and P3's three runs
# were 13/2/1 — a direction, not a conclusion. Pass "" to either to switch it off.

#
# v52: the other two upstream identity signals, each its own switch, ALL OFF by
# default until the planner rules on the v52 numbers (P3 stays the default):
#   -UserFrame              sentence naming both parties at the top of the system prompt
#   -UserPersonaPosition    head | tail  (where the -UserPersona paragraph goes)
#   -FewshotUserPrefix      on | off     (label the few-shot example user lines too)
#   -FewshotPrefixSkipNamed on | off     (leave an example line alone if it already names the user)
# Same values are switchable at runtime through POST /config.
#
# v57 (planner ruling 2026-09-09): -FewshotMessages is ON by default.
# It had always been off, which meant the v30 decision "few-shot travels as
# messages+boundary" (2026-08-22) was never deployed -- `fewshot_mode` only
# takes effect when this is on, and this script did not even expose it, so the
# proxy's own default of "off" always won. Meanwhile every measurement round
# from v50 to v54 set it on by hand. Measured single-variable in v56 (arm F):
# ACT coverage 23/40 -> 20/20 turns, leading marker 51 -> 27 characters
# (212 -> 103 ms), replies lost entirely 2/40 -> 0/20, end to end 2557 -> 2391 ms.
# Pass -FewshotMessages off to get the pre-v57 behaviour back.
#
# v53 (planner ruling 2026-09-09): -StripTools is ON by default. AIRI attaches 7 tool
# definitions to every request = 2622 of the 4952 prompt tokens production was sending
# (53%; `builtIn_emitSparkCommand` alone 1505). Measured on the real desktop path, six
# 20-turn runs per arm: with the tools forwarded she read User as a third party in 38 of
# 120 turns and invented a tool that does not exist five times (`<|CALL ["ai-assistant"…]|>`);
# with them stripped, 5/120 and 0. Per-turn latency is the same either way (the tools sit
# in the cached prefix). Nothing uses those tools today -- no MCP server is configured and
# no frontend module is connected -- so nothing is lost right now.
#   ⚠️ The cost is deferred, not absent: `body.pop("tools")` is all-or-nothing (no per-tool
#   filtering), so the day a real module connects, `builtIn_emitSparkCommand` (the Minecraft
#   line's channel) and `get_weather` go with it. That needs per-tool filtering first -- a
#   code change, recorded in ENVIRONMENT §10 item 20.
#   Pass -StripTools off to send them again (that is the pre-v53 production behaviour).

param(
    [ValidateSet('off', 'on')]
    [string]$Thinking = 'off',
    [int]$Port = 11435,
    [string]$Upstream = 'http://localhost:11434',
    [string]$ModelOverride = 'qwen3.5:4b',
    [ValidateSet('off', 'on')]
    [string]$StripTools = 'on',
    [ValidateSet('off', 'on')]
    [string]$FewshotMessages = 'on',
    [string]$UserPersona = "現在跟你講話的使用者就是$(if ($env:KAMIE_USER_NAME) { $env:KAMIE_USER_NAME } else { 'User' })本人。",
    [string]$UserNamePrefix = "$(if ($env:KAMIE_USER_NAME) { $env:KAMIE_USER_NAME } else { 'User' })：",
    [string]$UserFrame = '',
    [ValidateSet('head', 'tail')]
    [string]$UserPersonaPosition = 'tail',
    [ValidateSet('off', 'on')]
    [string]$FewshotUserPrefix = 'off',
    [ValidateSet('off', 'on')]
    [string]$FewshotPrefixSkipNamed = 'on',
    # v73 task D: the ~441 chars of artistry/image_journal/canvas widget instructions
    # AIRI appends after the card. OFF THE PROMPT SINCE v73 under the customer's
    # standing authorisation (00 v1.6 一, 2026-09-21: turn off add-on features that
    # get in the way of chatting, no need to ask). 🔴 SWITCH ONLY -- no code path is
    # deleted; `-StripWidget off` restores v72's byte-identical prompt.
    # NOT the same as StripInjections, which would ALSO take the 260-char
    # markdown/LaTeX prefix; that one stays off.
    [ValidateSet('off', 'on')]
    [string]$StripWidget = 'on',
    # v73b task 3: repair (or strip) the control markers the model writes malformed,
    # in-stream, so AIRI's reasoning filter cannot drop the rest of the reply.
    # End-to-end matrix (mainline-v73b): a dangling angle tag such as `<DELAY 1|>`
    # or `<description>` loses the WHOLE reply today (0 characters spoken) and is
    # fully recovered with this on; a healthy reply is byte-identical either way.
    # `-FixMarkers off` restores v73's behaviour exactly. No code path is deleted.
    [ValidateSet('off', 'on')]
    [string]$FixMarkers = 'on',
    # v61: where the card's 【輸出格式規則】 block goes. card = leave it in the card text
    # (production, byte-identical); short = ~170-char minimum; separate = own system
    # message after the few-shot boundary; off = removed. Measured in v61, NOT shipped:
    # the default stays 'card' until the planner rules on the numbers.
    [ValidateSet('card', 'short', 'separate', 'off')]
    [string]$FormatRules = 'card',
    # v80 (spec-v80 A-1): generation sampling. The proxy has read LLM_PROXY_SAMPLING
    # since v25, but this script never set it (found in v58), so production always
    # ran the Modelfile's values -- 1.0 / 0.95 / top_k 20 / presence 1.5, which is
    # Qwen's "thinking mode, general tasks" row -- with thinking switched off.
    #   on  = inject temperature and top_p (0.7 / 0.8 = Qwen's non-thinking row).
    #         presence_penalty is NOT sent, so the Modelfile's 1.5 still applies,
    #         and top_k stays 20: byte-for-byte the request of v79's St arm
    #         (customer, blind: 7/14 "可以" vs 2/14 for the Modelfile values).
    #   off = send nothing; the Modelfile values apply (production before v80).
    # Revert in one line: `-Sampling off`, or at runtime POST /config {"sampling":"off"}.
    # v80 A-3: the DEFAULT is now 'on', pending the customer's own chat (spec-v80 §0):
    # on the real desktop path (mainline-v80 A-2, 2 x 20 turns interleaved) every forwarded
    # payload carried 0.7/0.8 and no presence_penalty, ACT 20/20 vs 20/20, no silent turn,
    # first audio median 2438 vs 2665 ms (paired on-off median -138 ms). If he says it does
    # not work, change the next line back to 'off' -- that is the whole revert.
    [ValidateSet('off', 'on')]
    [string]$Sampling = 'on',
    [double]$SamplingTemperature = 0.7,
    [double]$SamplingTopP = 0.8,
    # v83 task B: the history layer -- how she sees her own earlier turns (see the
    # history_profile comment in reasoning_proxy.py). DEFAULT 'off' = byte-identical to v82
    # (spec-v83 B-3: switchable, default off, accepts a profile). The other presets are
    # measurement arms until the planner and the customer rule; a request header
    # `X-History-Profile: <name>` overrides this per request. POST /config
    # {"history_profile":"off"} switches back at runtime.
    [ValidateSet('off', 'record', 'gist', 'window2', 'recent1_gist', 'gist_inplace', 'custom')]
    [string]$HistoryProfile = 'off',
    # which model writes the one-line gists of her earlier lines (gist / recent1_gist only)
    [string]$HistoryGistModel = 'qwen3.5:4b',
    # v84 spec B-1 #4: a ceiling on ONE reply's length (max_tokens, chat requests only --
    # thinking-on requests are untouched; a caller's own limit wins). Resource-only: 512 is
    # 1.7x the longest qwen3.5:4b reply ever seen (298 tokens over 7,177 replies, v81-v83). A reply that
    # hits it is loud (/health reply_cap_hits, reply-cap-events.jsonl). '-ReplyCap off' =
    # the byte-identical v83 request.
    [ValidateSet('off', 'on')]
    [string]$ReplyCap = 'on',
    [int]$ReplyCapTokens = 512,
    # v89 B-1: two knobs that used to exist only at runtime (POST /config) now have a
    # start-time parameter too, because the shipped her needs them from the first request:
    # the few-shot transport (drop = take the card's demos out, do not re-insert -- the
    # fine-tuned models' training shape) and the blind-chat pad (0 = none).
    [ValidateSet('messages', 'messages+boundary', 'messages+boundary_user', 'messages+boundary_head',
                 'single_user', 'system_tagged', 'system_lines', 'drop')]
    [string]$FewshotMode = 'messages+boundary',
    [int]$PadMs = 0,
    # v89 B-2: where a thinking-on request goes while -ModelOverride is a model that cannot think
    # (the v87 fine-tunes return empty content with reasoning on). '' = do not reroute (v88 behaviour).
    # Counted on /health (think_routed) and written to think-route-events.jsonl.
    [string]$ThinkModel = ''
)

# v89 B-1: the default HER has ONE source -- character-cards\default-card.json, profiles[current]
# (model, few-shot transport, sampling values, history, reply cap, pad; the card is applied by
# start_all). A parameter given on the command line still wins. If the file cannot be read the
# literals above are used (= v85-v88's qwen3.5:4b + demos) and this says so in red; start_all then
# reads the running /config back against the file and reports NOT READY, so it cannot pass silently.
$herDefault = $null
try {
    . (Join-Path (Split-Path -Parent $PSScriptRoot) 'character-cards\default_card_lib.ps1')
    $herDefault = Get-HerProfile
}
catch { Write-Host "default-card.json unreadable, using the script literals (qwen3.5:4b + demos): $($_.Exception.Message)" -ForegroundColor Red }
if ($herDefault) {
    if (-not $PSBoundParameters.ContainsKey('ModelOverride')) { $ModelOverride = $herDefault.Model }
    if (-not $PSBoundParameters.ContainsKey('FewshotMode')) { $FewshotMode = $herDefault.FewshotMode }
    if (-not $PSBoundParameters.ContainsKey('SamplingTemperature')) { $SamplingTemperature = $herDefault.Temperature }
    if (-not $PSBoundParameters.ContainsKey('SamplingTopP')) { $SamplingTopP = $herDefault.TopP }
    if (-not $PSBoundParameters.ContainsKey('HistoryProfile')) { $HistoryProfile = $herDefault.HistoryProfile }
    if (-not $PSBoundParameters.ContainsKey('ReplyCapTokens')) { $ReplyCapTokens = $herDefault.ReplyCapTokens }
    if (-not $PSBoundParameters.ContainsKey('PadMs')) { $PadMs = $herDefault.PadMs }
    if (-not $PSBoundParameters.ContainsKey('ThinkModel')) { $ThinkModel = $herDefault.ThinkModel }
    Write-Host "default her = $($herDefault.Name) ($($herDefault.Model), fewshot_mode=$($herDefault.FewshotMode))"
}

$venv = if ($env:PYTHON_EXE) { $env:PYTHON_EXE } else { "python" }
$script = Join-Path $PSScriptRoot "reasoning_proxy.py"

$env:LLM_PROXY_THINKING = $Thinking
$env:LLM_PROXY_PORT = $Port
$env:LLM_PROXY_UPSTREAM = $Upstream
$env:LLM_PROXY_MODEL_OVERRIDE = $ModelOverride
$env:LLM_PROXY_STRIP_TOOLS = $StripTools
$env:LLM_PROXY_FEWSHOT_MESSAGES = $FewshotMessages
$env:LLM_PROXY_USER_PERSONA = $UserPersona
$env:LLM_PROXY_USER_NAME_PREFIX = $UserNamePrefix
$env:LLM_PROXY_USER_FRAME = $UserFrame
$env:LLM_PROXY_USER_PERSONA_POSITION = $UserPersonaPosition
$env:LLM_PROXY_FEWSHOT_USER_PREFIX = $FewshotUserPrefix
$env:LLM_PROXY_FEWSHOT_PREFIX_SKIP_NAMED = $FewshotPrefixSkipNamed
$env:LLM_PROXY_STRIP_WIDGET = $StripWidget
$env:LLM_PROXY_FIX_MARKERS = $FixMarkers
$env:LLM_PROXY_FORMAT_RULES = $FormatRules
# Invariant culture so a comma-decimal locale can never hand the proxy "0,7".
$inv = [Globalization.CultureInfo]::InvariantCulture
$env:LLM_PROXY_SAMPLING = $Sampling
$env:LLM_PROXY_SAMPLING_TEMPERATURE = $SamplingTemperature.ToString($inv)
$env:LLM_PROXY_SAMPLING_TOP_P = $SamplingTopP.ToString($inv)
$env:LLM_PROXY_HISTORY_PROFILE = $HistoryProfile
$env:LLM_PROXY_HISTORY_GIST_MODEL = $HistoryGistModel
$env:LLM_PROXY_REPLY_CAP = $ReplyCap
$env:LLM_PROXY_REPLY_CAP_TOKENS = $ReplyCapTokens
$env:LLM_PROXY_FEWSHOT_MODE = $FewshotMode
$env:LLM_PROXY_PAD_MS = $PadMs
$env:LLM_PROXY_THINK_MODEL = $ThinkModel
$env:PYTHONIOENCODING = "utf-8"

Write-Host "LLM proxy -> http://127.0.0.1:$Port  (thinking=$Thinking, fewshot_messages=$FewshotMessages, fewshot_mode=$FewshotMode, pad_ms=$PadMs, think_model=$ThinkModel, upstream=$Upstream, model_override=$ModelOverride, strip_tools=$StripTools, user_persona=$UserPersona, user_name_prefix=$UserNamePrefix, user_frame=$UserFrame, persona_position=$UserPersonaPosition, fewshot_user_prefix=$FewshotUserPrefix, skip_named=$FewshotPrefixSkipNamed, format_rules=$FormatRules, sampling=$Sampling $($env:LLM_PROXY_SAMPLING_TEMPERATURE)/$($env:LLM_PROXY_SAMPLING_TOP_P), history_profile=$HistoryProfile, reply_cap=$ReplyCap $ReplyCapTokens)"
& $venv $script
