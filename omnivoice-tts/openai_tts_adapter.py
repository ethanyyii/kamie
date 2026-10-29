"""OpenAI-compatible `/v1/audio/speech` front-end for OmniVoice. Production, port 5050.

Takes over from the CosyVoice2 adapter in `../cosyvoice-tts`, which stays on disk
(plus an explicit `openai_tts_adapter.BACKUP-v11-cosyvoice2.py`) so the old engine
can be brought back by starting it instead of this one. Nothing about the HTTP
shape changed, so AIRI needs no code change and no settings change: same port,
same `/v1/audio/speech`, same single wav body.

Why synchronous rather than chunked-streaming
---------------------------------------------
Two independent reasons, and either one alone would be enough. AIRI's
OpenAI-compatible path does `decodeAudioData` on the whole body, so it cannot
consume a chunked response; and OmniVoice is non-autoregressive, so the whole
utterance materialises at once and there would be no intermediate chunk to send
even if AIRI could take one.

AIRI still splits a reply into sentence-sized segments and calls this endpoint
once per segment, so segment 1 plays while segment 2 is still being made.
Per-segment latency is what matters.

Timing log
----------
Every request records four wall-clock instants -- received, synthesis started,
synthesis finished, response fully written -- into `timing.jsonl`, so an
end-to-end measurement can line these up against the browser's own timeline
without touching AIRI's source. `time.time()` on Windows only ticks every
~15.6ms, so each instant is anchored once per request and the rest are derived
from `perf_counter` deltas; offsets between the four points are therefore
sub-millisecond even though the anchor itself is not.

Settings (environment variables, all optional)
----------------------------------------------
  OMNIVOICE_NUM_STEP   diffusion steps, default 16. Synthesis time is a straight
                       line through the origin at ~32ms/step, so 16 halves it:
                       1.06s -> 0.53s for a mid-length sentence on this box.
                       32 remains selectable and costs nothing but time.
                       (Supersedes a v13-era note claiming 16 was rejected for
                       final-syllable drift. That was never the user's verdict;
                       on a real listen they judged 16 acceptable and steadier
                       than 32. 8 is untested in real use and may be better
                       still -- side-by-side is harsher than actual listening.
                       Re-listen after the Phase 1 voice-clone change, since
                       this was decided on the current reference voice.)
  OMNIVOICE_PORT       default 5050
  OMNIVOICE_ALIAS_PORTS
                       extra ports to serve the same app on, comma separated.
                       Only reason this exists: AIRI's saved speech Base URL
                       still points at 5051 (the v11 test adapter) and that
                       field could not be rewritten from automation, so 5051 is
                       served as an alias until the setting is updated by hand.
                       Production is 5050; drop this once the setting moves.
  OMNIVOICE_LANGUAGE   default "Chinese"
  OMNIVOICE_REF_WAV    reference clip for voice cloning
  OMNIVOICE_REF_TEXT   transcript of that clip
  OMNIVOICE_NORMALIZE  "1" to pass normalize_text=True, default off
  OMNIVOICE_LOG        timing log path, default ./timing.jsonl

`num_step` and `normalize_text` are also readable/writable at runtime via
`/config`, so they can be A/B'd inside AIRI without paying the model reload.

Call stack:

  POST /v1/audio/speech
    -> {@link synthesize}
      -> OmniVoice.generate
"""

import asyncio
import io
import json
import os
import re
import sys
import threading
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from starlette.background import BackgroundTask

from omnivoice import OmniVoice

HERE = Path(__file__).parent

# OmniVoice's own normalize_text=True path imports `tn.*` from WeTextProcessing,
# which cannot be built on Windows (pynini). `tn_compat` exposes the same module
# path over `wetext`, so putting it on sys.path lets the official code path run
# unmodified. Harmless when normalization stays off: nothing imports `tn` then.
sys.path.append(str(HERE / "tn_compat"))

SR = 24000
MODEL_ID = "k2-fsa/OmniVoice"

# Byte-identical copy of the clip CosyVoice2 shipped and was pinned to, so the
# voice the user approved in v11 is the voice this serves. Kept local to this
# folder rather than reached for across the tree.
# v33: the old default was CosyVoice's own demo asset (zero_shot_prompt.wav, MD5
# 1e8880cff3c491b2f1a901f253c1a063) — a performative read whose delivery every
# candidate inherited, and whose rights are unclear (that repo's README says some
# demo assets came from the internet). Replaced with a clip from MELI (UBC
# Mandarin-English Language Interview Corpus, CC BY 4.0, ethics board H23-03205,
# participants signed an audio release): spontaneous speech, headworn mic, 44.1kHz
# downsampled to 24k. See reference/ATTRIBUTION.md. The old file stays on disk so
# the previous voice remains one env var away.
REF_WAV = os.environ.get("OMNIVOICE_REF_WAV", str(HERE / "reference" / "meli-F02D-spontaneous-24k.wav"))
REF_TEXT = os.environ.get(
    "OMNIVOICE_REF_TEXT",
    (HERE / "reference" / "meli-F02D-spontaneous-24k.txt").read_text(encoding="utf-8").strip(),
)

LANGUAGE = os.environ.get("OMNIVOICE_LANGUAGE", "Chinese")
LOG_PATH = Path(os.environ.get("OMNIVOICE_LOG", str(HERE / "timing.jsonl")))

app = FastAPI(title="OmniVoice OpenAI-compatible adapter")

# AIRI runs in a browser, so the speech call is a cross-origin XHR from the Stage origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
# The reference clip encoded once at startup. Re-deriving it per request is the
# 0.3s CosyVoice2 used to throw away, and the spec is explicit about not
# repeating that.
_prompt = None
# generate() holds per-call state on the model, so overlapping requests would
# interleave. AIRI sends one segment at a time anyway.
_lock = threading.Lock()

_settings = {
    "num_step": int(os.environ.get("OMNIVOICE_NUM_STEP", "16")),
    "normalize_text": os.environ.get("OMNIVOICE_NORMALIZE", "0") == "1",
    # Safety valve, not an optimisation. OmniVoice's PyTorch allocator grows to
    # fit the longest utterance this process has ever synthesised and never
    # gives it back, so a single freak input permanently raises the VRAM floor
    # for the rest of the session: one 30s sentence measured 12091 MiB of 12282.
    #
    # AIRI's own chunker caps a segment at 12 words, so real traffic sits far
    # below this; 200 characters only catches something that has already gone
    # wrong upstream.
    "max_input_chars": int(os.environ.get("OMNIVOICE_MAX_INPUT_CHARS", "200")),
    # What to do when the cap is exceeded:
    #   "split"    synthesise in pieces and concatenate  (default)
    #   "truncate" synthesise the first N characters, drop the rest
    #   "error"    reject with 413
    # `split` is the default because it is the only one that neither loses words
    # nor fails the caller; the cost is a seam in the prosody at each cut.
    "over_limit": os.environ.get("OMNIVOICE_OVER_LIMIT", "split"),
    # Refuse to read stream-control markers and JSON debris out loud. See
    # `_looks_like_marker_debris`. Switchable because it is a heuristic sitting
    # in front of the only component that can embarrass her audibly.
    "filter_markers": os.environ.get("OMNIVOICE_FILTER_MARKERS", "1") not in ("0", "false", "False"),
    # v31: equalise loudness across segments. OmniVoice scales output volume to the
    # reference clip's RMS (0.0725 here -> quiet output), and short segments come out
    # with wildly different levels (v31 measured RMS spreads of 128-496% across takes
    # of the same two-character utterance). When on, each response is linearly gained
    # toward `target_rms`, capped so the peak never exceeds 0.95 — no clipping, no
    # waveshaping. Off by default: the client A/Bs it before it becomes production.
    "normalize_loudness": os.environ.get("OMNIVOICE_NORMALIZE_LOUDNESS", "0") == "1",
    "target_rms": float(os.environ.get("OMNIVOICE_TARGET_RMS", "0.10")),
    # v31: convert the synthesizer's input to Simplified glyphs. The model was
    # trained on Simplified text and misreads several Traditional-only glyphs
    # (measured: 誰/錶 fail 0/6 as-is, 5-6/6 after conversion). Off by default
    # until the client A/Bs it.
    "t2s_input": os.environ.get("OMNIVOICE_T2S_INPUT", "0") == "1",
}

_t2s_conv = None

# opencc's t2s misses these glyphs (v34d: 著→着 bit us twice). Keep in sync with
# tts-v35\glyph_v35.py EXTRA — that file is the source of truth for audition paths;
# this copy exists so the production service has no import outside its own folder.
_T2S_EXTRA = {"著": "着"}


def _t2s():
    global _t2s_conv
    if _t2s_conv is None:
        from opencc import OpenCC
        _t2s_conv = OpenCC("t2s")
    return _t2s_conv

# Prefer to cut where a listener already expects a pause.
_SPLIT_AFTER = "。！？!?…；;，,、"

_req_seq = 0


class SpeechRequest(BaseModel):
    """Subset of OpenAI's `/v1/audio/speech` body that this adapter honours.

    `voice` is accepted and ignored: this build is pinned to the one reference
    clip above, matching how the CosyVoice2 adapter behaved.
    """

    model: str = "OmniVoice"
    input: str
    voice: str = ""
    response_format: str = "wav"
    speed: float = 1.0


class Clock:
    """Wall-clock instants that keep `perf_counter` precision between them.

    Windows' `time.time()` granularity is ~15.6ms, which is coarse enough to
    smear the four per-request instants into each other. Anchoring once and
    adding monotonic deltas keeps the spacing exact; only the shared anchor
    carries the coarse absolute error.
    """

    def __init__(self):
        self._wall0 = time.time()
        self._perf0 = time.perf_counter()

    def now(self) -> float:
        return self._wall0 + (time.perf_counter() - self._perf0)


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts).isoformat(timespec="milliseconds")


def _load():
    global _model, _prompt
    if _model is None:
        print(f"[omnivoice] loading {MODEL_ID} ...", flush=True)
        t0 = time.perf_counter()
        _model = OmniVoice.from_pretrained(MODEL_ID, device_map="cuda:0", dtype=torch.float16)
        _model.eval()
        t_load = time.perf_counter() - t0

        t0 = time.perf_counter()
        _prompt = _model.create_voice_clone_prompt(ref_audio=REF_WAV, ref_text=REF_TEXT)
        t_prompt = time.perf_counter() - t0

        # A first real generate() pays for CUDA kernel autotuning and lazy module
        # init; without it request #1 is several times slower than the steady
        # state and every later number is polluted by comparison.
        t0 = time.perf_counter()
        _model.generate(
            text="你好，我準備好了。",
            language=LANGUAGE,
            voice_clone_prompt=_prompt,
            num_step=_settings["num_step"],
        )
        t_warm = time.perf_counter() - t0

        print(
            f"[omnivoice] load {t_load:.1f}s | voice-clone prompt {t_prompt:.2f}s (cached) "
            f"| warmup {t_warm:.2f}s | num_step={_settings['num_step']}",
            flush=True,
        )
    return _model


def _wav_bytes(audio: np.ndarray) -> bytes:
    """Pack a float32 [-1, 1] mono array into a 16-bit PCM wav container."""
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def _write_timing(record: dict):
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": "OmniVoice", "object": "model", "owned_by": "local"}]}


@app.get("/health")
def health():
    return {"status": "ok", "loaded": _model is not None, **_settings}


@app.get("/config")
def get_config():
    return {**_settings, "language": LANGUAGE, "ref_wav": REF_WAV, "sample_rate": SR}


@app.post("/config")
def set_config(patch: dict):
    """Flip `num_step` / `normalize_text` without paying the model reload.

    Both stay switchable on purpose: the spec wants step 32 as the shipped
    default but step 16 kept one keystroke away for the user's own A/B.
    """
    for key in ("num_step", "normalize_text", "max_input_chars", "over_limit", "filter_markers",
                "normalize_loudness", "target_rms", "t2s_input"):
        if key in patch:
            _settings[key] = patch[key]
    print(f"[omnivoice] config -> {_settings}", flush=True)
    return _settings


@app.post("/debug/empty_cache")
def empty_cache():
    """Hand torch's unused reserved VRAM back to the driver, and report whether
    that was worth anything.

    torch keeps freed blocks in its own pool rather than returning them, so
    `reserved - allocated` is memory this process is holding but not using.
    `empty_cache()` releases exactly that and nothing else -- if the gap is
    already small the call is a no-op, which is the answer v16 wanted and never
    got. It cannot help with the failure that actually cost this project 6.5GB:
    orphaned llama-server processes are separate processes, and nothing inside
    this one can reach them.

    Left as a debug endpoint rather than wired into synthesis: freeing the pool
    means the next request has to reacquire from the driver, which is a cost
    paid on the latency path for a benefit nobody has shown exists.
    """
    import torch
    if not torch.cuda.is_available():
        return {"error": "no CUDA device"}
    mb = lambda n: round(n / 2**20, 1)  # noqa: E731
    before = {"reserved": torch.cuda.memory_reserved(),
              "allocated": torch.cuda.memory_allocated()}
    torch.cuda.empty_cache()
    after = {"reserved": torch.cuda.memory_reserved(),
             "allocated": torch.cuda.memory_allocated()}
    freed = before["reserved"] - after["reserved"]
    out = {
        "reserved_mb_before": mb(before["reserved"]),
        "reserved_mb_after": mb(after["reserved"]),
        "allocated_mb_before": mb(before["allocated"]),
        "allocated_mb_after": mb(after["allocated"]),
        "reclaimable_mb_before": mb(before["reserved"] - before["allocated"]),
        "freed_mb": mb(freed),
    }
    print(f"[omnivoice] empty_cache freed {out['freed_mb']} MiB", flush=True)
    return out


# Structural JSON punctuation that never appears in speech but dominates a
# leaked marker payload. `":` specifically is a JSON key terminator -- Chinese
# text uses the fullwidth colon, and English prose puts a space between the
# colon and the quote, so neither collides with this.
_JSON_KEY = '":'
_MARKER_PREFIX = re.compile(r'^(ACT|DELAY|CALL)\b\s*[{\[]', re.IGNORECASE)


def _looks_like_marker_debris(text: str) -> bool:
    """Is this a leaked stream-control marker rather than something to say?

    AIRI's marker parser only recognises `<|ACT ...|>` when both delimiters are
    present. When the model emits the payload without them the blob is treated
    as ordinary speech and arrives here, and has been synthesized aloud in
    fragments, because the sentence chunker splits on the `:` and `,` inside the
    JSON.

    Deliberately conservative: a false positive silences something she meant to
    say, which is worse than the problem being solved. The named-marker branch
    requires a marker keyword immediately followed by its payload bracket; the
    fragment branch additionally requires a JSON key terminator, which ordinary
    prose does not contain.

    Debris seen in production, all caught here:
        'ACT {"emotion":'
        '{"name":'
        '"angry","intensity":1},"motion":"face-palm"}'
    """
    t = text.strip()
    if not t:
        return False
    # A marker name followed by its payload, with or without a complete body.
    if _MARKER_PREFIX.match(t):
        return True
    # A bare JSON fragment left over from a payload split across chunks.
    if _JSON_KEY not in t:
        return False
    return t[0] in '{["'


def _silent_wav(sample_rate: int) -> bytes:
    """A well-formed, empty wav.

    Returned instead of an HTTP error so a filtered fragment stays invisible to
    AIRI: an error status would surface in the chat as a failed speech request,
    which is a different way of letting the leak reach the user.
    """
    import struct
    return (b'RIFF' + struct.pack('<I', 36) + b'WAVEfmt '
            + struct.pack('<IHHIIHH', 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
            + b'data' + struct.pack('<I', 0))


def _split_for_limit(text: str, limit: int) -> list[str]:
    """Cut `text` into pieces of at most `limit` characters.

    Cuts after punctuation where possible so the seam lands where a listener
    already expects a pause; falls back to a hard cut only when a run of `limit`
    characters contains none.
    """
    pieces, rest = [], text
    while len(rest) > limit:
        window = rest[:limit]
        cut = max((window.rfind(c) for c in _SPLIT_AFTER), default=-1)
        # Refuse a cut so early that it would make many tiny fragments.
        cut = cut + 1 if cut >= limit // 3 else limit
        pieces.append(rest[:cut].strip())
        rest = rest[cut:].lstrip()
    if rest:
        pieces.append(rest)
    return [p for p in pieces if p]


@app.post("/v1/audio/speech")
def synthesize(req: SpeechRequest):
    global _req_seq

    clock = Clock()
    t_recv = clock.now()

    text = (req.input or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="input is required")

    # v31: OmniVoice garbles Traditional-only glyphs (誰→"當然", 錶 dropped — trad 0/6
    # vs simp 6/6 on the same sentences). The audience hears sounds, not glyphs, so
    # feeding the model its native script costs nothing. Text shown in the UI is
    # untouched — this rewrites only what the synthesizer reads.
    if _settings.get("t2s_input"):
        text = _t2s().convert(text)
        for _k, _v in _T2S_EXTRA.items():
            text = text.replace(_k, _v)

    if _settings.get("filter_markers") and _looks_like_marker_debris(text):
        print(f"[omnivoice] refused to speak marker debris: {text[:60]!r}", flush=True)
        return Response(content=_silent_wav(SR), media_type="audio/wav")

    limit = int(_settings["max_input_chars"])
    over = len(text) > limit > 0
    mode = _settings["over_limit"]
    if over and mode == "error":
        raise HTTPException(
            status_code=413,
            detail=f"input is {len(text)} chars, over the {limit}-char cap "
                   f"(over_limit=error; set /config over_limit to split or truncate)",
        )
    if over and mode == "truncate":
        text = text[:limit]
    parts = _split_for_limit(text, limit) if (over and mode == "split") else [text]

    with _lock:
        model = _load()
        _req_seq += 1
        seq = _req_seq

        # OpenAI's default speed is 1.0; OmniVoice treats an explicit speed as a
        # duration constraint, so only forward it when it actually asks for a
        # change.
        extra = {} if abs(req.speed - 1.0) < 1e-6 else {"speed": req.speed}

        t_synth_start = clock.now()
        chunks = []
        for part in parts:
            out = model.generate(
                text=part,
                language=LANGUAGE,
                voice_clone_prompt=_prompt,
                num_step=_settings["num_step"],
                normalize_text=_settings["normalize_text"],
                **extra,
            )
            chunks.append(out[0])
        t_synth_end = clock.now()

        audio = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)

        if _settings.get("normalize_loudness"):
            rms = float(np.sqrt((audio ** 2).mean())) if len(audio) else 0.0
            peak = float(np.abs(audio).max()) if len(audio) else 0.0
            if rms > 1e-4 and peak > 0:
                # linear gain toward target_rms, but never let the peak reach 0.95:
                # loudness is equalised as far as headroom allows, with zero clipping.
                gain = min(float(_settings["target_rms"]) / rms, 0.95 / peak)
                audio = audio * gain

        wav = _wav_bytes(audio)

    audio_s = len(audio) / SR
    synth_s = t_synth_end - t_synth_start
    print(
        f"[omnivoice] #{seq} {len(text)} chars -> {audio_s:.2f}s audio in {synth_s:.2f}s "
        f"(RTF {synth_s / audio_s:.2f}) step={_settings['num_step']}",
        flush=True,
    )

    def _log_sent():
        # BackgroundTask runs after the response body has been handed to the
        # transport, which is the closest this side can get to "the browser has
        # it" without instrumenting AIRI.
        t_sent = clock.now()
        _write_timing({
            "seq": seq,
            "chars": len(text),
            "text": text,
            "parts": len(parts),
            "over_limit": over,
            "audio_s": round(audio_s, 3),
            "num_step": _settings["num_step"],
            "normalize_text": _settings["normalize_text"],
            "wav_bytes": len(wav),
            "t_recv": t_recv,
            "t_synth_start": t_synth_start,
            "t_synth_end": t_synth_end,
            "t_sent": t_sent,
            "iso_recv": _iso(t_recv),
            "iso_synth_start": _iso(t_synth_start),
            "iso_synth_end": _iso(t_synth_end),
            "iso_sent": _iso(t_sent),
            "queue_s": round(t_synth_start - t_recv, 4),
            "synth_s": round(synth_s, 4),
            "write_s": round(t_sent - t_synth_end, 4),
            "total_s": round(t_sent - t_recv, 4),
        })

    return Response(content=wav, media_type="audio/wav", background=BackgroundTask(_log_sent))


async def _serve(ports: list[int]):
    """Serve the one loaded model on several ports from a single process.

    Two uvicorn servers rather than two processes: the model is 3.3GB of VRAM
    and must not be loaded twice.
    """
    servers = [
        uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=p, log_level="warning"))
        for p in ports
    ]
    await asyncio.gather(*(s.serve() for s in servers))


if __name__ == "__main__":
    port = int(os.environ.get("OMNIVOICE_PORT", "5050"))
    alias = [int(p) for p in os.environ.get("OMNIVOICE_ALIAS_PORTS", "").split(",") if p.strip()]
    ports = [port, *alias]

    _load()
    print(
        f"[omnivoice] ready on {', '.join(f'http://127.0.0.1:{p}' for p in ports)}"
        f"  (timing log: {LOG_PATH})",
        flush=True,
    )
    sys.stdout.flush()
    asyncio.run(_serve(ports))
