# omnivoice-tts — production TTS (v12)

OmniVoice replaced CosyVoice2 as the production TTS in v12. This folder holds the
adapter; the Python environment is still `..\tts-test\omnivoice\.venv` (built in
v8 — moving a venv is a needless risk, so `start_tts.ps1` just points at it).

## Start

```powershell
cd omnivoice-tts
.\start_tts.ps1
```

Options: `-NumStep 16` (faster, the user heard the final syllable drift),
`-Normalize` (text normalization on), `-AliasPorts ""` (drop the 5051 alias once
AIRI's setting points at 5050).

## Ports

| Port | What |
|---|---|
| 5050 | production, the slot CosyVoice2 used to hold |
| 5051 | **temporary alias** of the same model — AIRI's saved speech Base URL still says 5051 and that settings field could not be rewritten from automation. Change AIRI to 5050 and this can go. |

## Switchable at runtime

`num_step` and `normalize_text` can be changed without paying the model reload:

```
curl http://127.0.0.1:5050/config
curl -X POST http://127.0.0.1:5050/config -H "Content-Type: application/json" -d "{\"num_step\":16}"
```

Shipped defaults: `num_step=32`, `normalize_text=false`.

## Timing log

Every request appends one line to `timing.jsonl` with four wall-clock instants
(received / synthesis start / synthesis end / response written). That is what the
v12 end-to-end measurement was aligned against.

## Rolling back to CosyVoice2

Nothing was removed. Stop this adapter and start the old one:

```powershell
cd <path-to-your-cosyvoice-clone>
.\.venv\Scripts\python.exe openai_tts_adapter.py
```

`cosyvoice-tts\openai_tts_adapter.BACKUP-v11-cosyvoice2.py` is a byte-identical
copy of that file as it stood on 8/2 22:43, kept in case the working copy is ever
edited.

## tn_compat

Text normalization needs WeTextProcessing, which cannot be built on Windows
(`pynini` feeds GCC flags to MSVC). `tn_compat/` maps the module path OmniVoice
imports onto `wetext`, the same team's pynini-free build — the one CosyVoice2 has
been using on this machine all along. See `tn_compat/README.md`.
