"""OpenAI-compatible `/v1/audio/speech` front-end for GLM-TTS 1.5B. Production candidate (v50), port 5050.

Same HTTP shape as the OmniVoice adapter in `../omnivoice-tts` (same port, same
`/v1/audio/speech`, single wav body, same `/health` `/config` `/debug/empty_cache`,
same four-instant timing log), so AIRI needs no code change and no settings
change to speak with this engine instead: start this script rather than that
one. Nothing on disk is removed or rewired -- the OmniVoice path is one start
script away (ENVIRONMENT.md section 4).

Why a separate process rather than an `engine=` switch inside the old adapter
------------------------------------------------------------------------------
The two engines cannot share a venv (torch 2.3.1+cu121 pinned here versus
2.8+cu128 there, ENVIRONMENT section 8 "glm venv"), and both together do not fit
the 12 GB card next to the LLM. So the seam is the HTTP contract, not a Python
import, and switching engines is "start the other script".

What is a parameter and what is not (00 section 8, reversibility)
-----------------------------------------------------------------
Everything that decides the voice is a setting, never a constant in this file:
which prompt clip (`prompt` profile: young2 = the chosen voice, gprobe = the
fallback the client named), its transcript, the sampling seed, whether head
silence is cut, whether punctuation is softened, whether a suspected voice jump
is regenerated. Switching to the fallback voice is one `/config` call and takes
about two seconds (prompt re-encoding); it does not need a restart.

The praat transform that produced young2 (formant x1.15, pitch x1.22) is NOT
applied here at request time -- it is a one-off transform of the prompt clip,
reproduced by `tools/make_prompt_v50.py` from parameters. The adapter consumes
whatever clip a profile names. See `prompts/PROVENANCE.md`.

Engine facts that shape the latency (v47 investigation)
-------------------------------------------------------
GLM has no token-level streaming API, so for a short segment the time to first
byte equals the time to synthesise the whole segment: LLM (autoregressive, ~50
tok/s here, 25 tokens per second of audio) plus flow/vocoder. AIRI splits a reply
into sentence-sized segments and calls this endpoint once per segment, so
per-segment latency is what the user waits for.

Settings (environment variables, all optional; most also live on `/config`)
---------------------------------------------------------------------------
  GLM_PORT                    default 5050
  GLM_ALIAS_PORTS             extra ports, comma separated (same reason as the OmniVoice one)
  GLM_REPO                    the official repo checkout (ckpt/ and frontend/ are relative to it)
  GLM_PROMPTS_DIR             folder holding profiles.json and the prompt clips
  GLM_PROMPT                  profile name, default "young2"
  GLM_SEED                    default 0 (v47/v48 packs the client heard were seed 0)
  GLM_LLM_DTYPE               fp16 (default; the one deviation from the official fp32,
                              needed on 12 GB, upstream issue #51 precedent) or fp32
  GLM_UNLOAD_SPEECH_TOKENIZER 1 (default) drops the 1.6 GB speech tokenizer after the
                              prompt is encoded; it is only needed to encode a prompt and
                              is reloaded on the fly when the prompt changes
  GLM_TRIM_HEAD               1 (default) cut leading silence; GLM_TRIM_HEAD_DB (-40),
                              GLM_TRIM_KEEP_MS (40)
  GLM_PUNCT_DOWNGRADE         0 (default); 1 turns mid-segment 。！？ into ， so the model
                              places shorter pauses (the client has not heard this yet)
  GLM_JUMP_DETECT             1 (default) measure the onset-vs-rest pitch/level signature of
                              a voice jump on every take and log loudly when it matches
  GLM_JUMP_REROLL             0 (default); 1 regenerates a flagged take with seed+1 before
                              answering (costs one extra synthesis on ~1/12 requests)
  GLM_T2S_INPUT               1 (default): Traditional -> Simplified glyphs for the
                              synthesiser only (the GLM packs the client judged were all t2s)
  GLM_MAX_INPUT_CHARS / GLM_OVER_LIMIT / GLM_FILTER_MARKERS / GLM_NORMALIZE_LOUDNESS /
  GLM_TARGET_RMS              same meaning as the OmniVoice adapter
  GLM_LOG                     timing log path, default ./timing.jsonl

Call stack:

  POST /v1/audio/speech
    -> {@link synthesize}
      -> {@link _generate}  (official generate_long, one fresh cache per request)
      -> audio_post.trim_head / onset_stats
"""

import asyncio
import gc
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = Path(os.environ.get("GLM_REPO", str(ROOT / "tts-v47" / "glm-repo"))).resolve()
PROMPTS_DIR = Path(os.environ.get("GLM_PROMPTS_DIR", str(HERE / "prompts"))).resolve()
LOG_PATH = Path(os.environ.get("GLM_LOG", str(HERE / "timing.jsonl"))).resolve()

# The official code reaches ckpt/ and frontend/ by relative path, so the process
# must live in the repo directory. Everything of ours is resolved to absolute
# paths above, before the chdir.
os.chdir(str(REPO))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import Response  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from starlette.background import BackgroundTask  # noqa: E402
from transformers import LlamaForCausalLM  # noqa: E402

from audio_post import head_silence_s, jump_suspect, normalize_loudness, onset_stats, trim_head  # noqa: E402
from cosyvoice.cli.frontend import SpeechTokenizer  # noqa: E402
from glmtts_inference import (  # noqa: E402
    DEVICE, generate_long, get_special_token_ids, load_frontends, local_flow_forward, local_llm_forward)
from llm.glmtts import GLMTTS  # noqa: E402
from utils import tts_model_util, yaml_util  # noqa: E402

SR = 24000
ENGINE = "GLM-TTS-1.5B"

app = FastAPI(title="GLM-TTS OpenAI-compatible adapter")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _env_flag(name: str, default: str) -> bool:
    return os.environ.get(name, default) not in ("0", "false", "False", "off", "")


_settings = {
    "prompt": os.environ.get("GLM_PROMPT", "young2"),
    "seed": int(os.environ.get("GLM_SEED", "0")),
    "unload_speech_tokenizer": _env_flag("GLM_UNLOAD_SPEECH_TOKENIZER", "1"),
    "trim_head": _env_flag("GLM_TRIM_HEAD", "1"),
    "trim_head_db": float(os.environ.get("GLM_TRIM_HEAD_DB", "-40")),
    "trim_keep_ms": float(os.environ.get("GLM_TRIM_KEEP_MS", "40")),
    "punct_downgrade": _env_flag("GLM_PUNCT_DOWNGRADE", "0"),
    "jump_detect": _env_flag("GLM_JUMP_DETECT", "1"),
    "jump_reroll": _env_flag("GLM_JUMP_REROLL", "0"),
    "jump_st_thr": float(os.environ.get("GLM_JUMP_ST_THR", "-1.0")),
    "jump_db_thr": float(os.environ.get("GLM_JUMP_DB_THR", "-1.5")),
    "t2s_input": _env_flag("GLM_T2S_INPUT", "1"),
    "max_input_chars": int(os.environ.get("GLM_MAX_INPUT_CHARS", "200")),
    "over_limit": os.environ.get("GLM_OVER_LIMIT", "split"),
    "filter_markers": _env_flag("GLM_FILTER_MARKERS", "1"),
    "normalize_loudness": _env_flag("GLM_NORMALIZE_LOUDNESS", "0"),
    "target_rms": float(os.environ.get("GLM_TARGET_RMS", "0.10")),
    # v55: the flow solver's two cost knobs. t_flow is a flat ~0.457 s on every
    # segment regardless of length, and it is 10 Euler steps x 2 estimator passes
    # (classifier-free guidance). Defaults here reproduce upstream exactly, and at
    # the defaults the original call path is what runs.
    "flow_steps": int(os.environ.get("GLM_FLOW_STEPS", "10")),
    "flow_cfg_rate": float(os.environ.get("GLM_FLOW_CFG_RATE", "-1")),
    # v55: the flow half and the vocoder are loaded with no dtype argument
    # (utils/yaml_util.py load_flow_model: .to(device).eval()), so they run in the
    # checkpoint's fp32. PyTorch leaves TF32 matmul OFF by default, and neither the
    # official repo nor this adapter ever turns it on -- so "official fp32" on an Ada
    # card means true IEEE fp32 in the DiT. This exposes the switch; it changes
    # numerics slightly, so it is a listening question, not a free win.
    "tf32": _env_flag("GLM_TF32", "0"),
    # v55: replace upstream's per-token nucleus sampler with a vectorised, semantics-
    # preserving one (see this file's header for what the original does per token).
    "fast_sampler": _env_flag("GLM_FAST_SAMPLER", "0"),
}
FLOW_STEPS_DEFAULT = 10
LLM_DTYPE = os.environ.get("GLM_LLM_DTYPE", "fp16")

_CONFIG_KEYS = tuple(_settings.keys())

# ---- glyph rule: copy of tts-v35\glyph_v35.py (t2s + the 著→着 hole), with its
# known-answer test run at startup so a broken opencc is loud, not silent.
_T2S_EXTRA = {"著": "着"}
_t2s_conv = None


def _install_fast_sampler() -> bool:
    """Swap cosyvoice.utils.common.nucleus_sampling for a vectorised equivalent.

    ras_sampling calls nucleus_sampling by module-global name, so replacing the module
    attribute is enough and nothing else in the checkout is touched. The original is kept
    and dispatched to whenever the setting is off, so this is one flag, not a fork.
    """
    try:
        from cosyvoice.utils import common as _cvc
    except Exception as e:  # pragma: no cover - loud, never silent
        print(f"[glm] fast_sampler unavailable: {e}", flush=True)
        return False
    if getattr(_cvc, "_v55_orig_nucleus", None) is not None:
        return True
    _orig = _cvc.nucleus_sampling
    _cvc._v55_orig_nucleus = _orig

    def _nucleus(weighted_scores, top_p=0.8, top_k=25, temperature=1.0):
        if not _settings["fast_sampler"]:
            return _orig(weighted_scores, top_p=top_p, top_k=top_k, temperature=temperature)
        probs = (weighted_scores / temperature).softmax(dim=0)
        v, idx = probs.topk(top_k)                 # already descending
        # the original keeps an entry while the cumulative probability BEFORE it is < top_p
        excl = torch.cumsum(v, dim=0) - v
        keep = excl < top_p
        keep[0] = True                              # the loop always takes at least one
        prob, indices = v[keep], idx[keep]
        return indices[prob.multinomial(1, replacement=True)]

    _cvc.nucleus_sampling = _nucleus
    print("[glm] fast_sampler installed (vectorised nucleus sampling; dispatches to the "
          "original whenever the flag is off)", flush=True)
    return True


def _t2s(text: str) -> str:
    global _t2s_conv
    if _t2s_conv is None:
        from opencc import OpenCC  # missing package must raise here, never fall through
        _t2s_conv = OpenCC("t2s")
    out = _t2s_conv.convert(text)
    for k, v in _T2S_EXTRA.items():
        out = out.replace(k, v)
    return out


def _glyph_self_test():
    cases = {"腳踝擋著誰的錶": "脚踝挡着谁的表", "辯解": "辩解", "你敢？": "你敢？", "我沒鐘錶": "我没钟表"}
    for src, want in cases.items():
        got = _t2s(src)
        if got != want:
            raise AssertionError(f"glyph self_test FAILED: {src!r} -> {got!r}, expected {want!r}")


# ---- marker-debris filter and length cap: verbatim behaviour of the OmniVoice adapter
_JSON_KEY = '":'
_MARKER_PREFIX = re.compile(r'^(ACT|DELAY|CALL)\b\s*[{\[]', re.IGNORECASE)
_SPLIT_AFTER = "。！？!?…；;，,、"
_MID_PUNCT = re.compile(r"[。！？!?；;](?=.)")


def _looks_like_marker_debris(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    if _MARKER_PREFIX.match(t):
        return True
    if _JSON_KEY not in t:
        return False
    return t[0] in '{["'


def _silent_wav(sample_rate: int) -> bytes:
    import struct
    return (b'RIFF' + struct.pack('<I', 36) + b'WAVEfmt '
            + struct.pack('<IHHIIHH', 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
            + b'data' + struct.pack('<I', 0))


def _split_for_limit(text: str, limit: int) -> list[str]:
    pieces, rest = [], text
    while len(rest) > limit:
        window = rest[:limit]
        cut = max((window.rfind(c) for c in _SPLIT_AFTER), default=-1)
        cut = cut + 1 if cut >= limit // 3 else limit
        pieces.append(rest[:cut].strip())
        rest = rest[cut:].lstrip()
    if rest:
        pieces.append(rest)
    return [p for p in pieces if p]


def _wav_bytes(audio: np.ndarray) -> bytes:
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


class Clock:
    """Wall-clock instants with perf_counter spacing (Windows time.time ticks every ~15.6 ms)."""

    def __init__(self):
        self._wall0 = time.time()
        self._perf0 = time.perf_counter()

    def now(self) -> float:
        return self._wall0 + (time.perf_counter() - self._perf0)


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts).isoformat(timespec="milliseconds")


def _write_timing(record: dict):
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _vram() -> dict:
    if not torch.cuda.is_available():
        return {}
    mb = lambda n: round(n / 2**20)  # noqa: E731
    return {"vram_reserved_mb": mb(torch.cuda.memory_reserved()),
            "vram_allocated_mb": mb(torch.cuda.memory_allocated()),
            "vram_max_reserved_mb": mb(torch.cuda.max_memory_reserved())}


# ---------------------------------------------------------------------------
# Model state
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_models = {}          # frontend, text_frontend, llm, token2wav, speech_tokenizer (or None)
_prompt = {}          # the encoded prompt currently in use
_profiles = {}
_counters = {"requests": 0, "jump_suspects": 0, "jump_rerolls": 0, "jump_unmeasurable": 0}
_req_seq = 0

# v84 spec F-1 (principle 10): the engine going bad must be LOUD. v83 saw CUDA break
# after a one-character utterance (cuDNN EXECUTION_FAILED, then every request OOM)
# while `/health` kept answering "ok" and she replied in text with no voice. The
# route below counts engine failures (not bad input: 400/413 are the caller's), and
# /health turns "broken" with the reasons -- start_all's check (item 16) and
# open_her read it. A CUDA-class error poisons the process until restart, so that
# reason is sticky; the others clear themselves.
_engine = {"synth_ok": 0, "synth_failures": 0, "synth_failures_consecutive": 0, "cuda_error_since_start": False,
           "slow_short_utterances": 0, "last_failure": None, "last_slow": None}
_CUDA_ERR = re.compile(r"CUDA|cuDNN|CUBLAS|out of memory|device-side|illegal memory|NCCL", re.I)
BROKEN_CONSECUTIVE = 2          # two engine failures in a row = broken (one can be a fluke)
SLOW_SHORT_CHARS, SLOW_SHORT_S = 10, 10.0   # v83: a 1-char segment took 40 s while Windows borrowed system RAM


def _card_total_mb():
    try:
        return round(torch.cuda.get_device_properties(0).total_memory / 2**20)
    except Exception:
        return None


def _engine_verdict() -> dict:
    reasons = []
    if _engine["cuda_error_since_start"]:
        reasons.append("cuda_error_since_start")
    if _engine["synth_failures_consecutive"] >= BROKEN_CONSECUTIVE:
        reasons.append("consecutive_synth_failures")
    total = _card_total_mb()
    v = _vram()
    if total and v.get("vram_reserved_mb", 0) > total:
        reasons.append("reserved_exceeds_card")        # Windows is lending system RAM right now
    warnings = []
    if total and v.get("vram_max_reserved_mb", 0) > total:
        warnings.append("reserved_exceeded_card_once")
    if _engine["slow_short_utterances"]:
        warnings.append("slow_short_utterances")
    return {"status": "broken" if reasons else "ok", "tts_broken_reasons": reasons, "tts_warnings": warnings,
            "card_total_mb": total, **_engine}


def _load_profiles() -> dict:
    p = PROMPTS_DIR / "profiles.json"
    profiles = json.loads(p.read_text(encoding="utf-8"))
    for name, prof in profiles.items():
        wav = Path(prof["wav"])
        if not wav.is_absolute():
            wav = PROMPTS_DIR / wav
        if not wav.exists():
            raise FileNotFoundError(f"prompt profile {name!r}: {wav} does not exist")
        prof["wav_abs"] = str(wav)
    return profiles


def _load_speech_tokenizer():
    m, fe = yaml_util.load_speech_tokenizer(os.path.join("ckpt", "speech_tokenizer"))
    return SpeechTokenizer(m, fe)


def _drop_speech_tokenizer():
    """Free the speech tokenizer (1.6 GB) -- it is used only to encode a prompt clip."""
    st = _models.get("speech_tokenizer")
    if st is None:
        return 0
    before = torch.cuda.memory_reserved()
    alloc_before = torch.cuda.memory_allocated()
    _models["frontend"].speech_tokenizer = None
    _models["speech_tokenizer"] = None
    # Break the object's own references too, so nothing that still points at the
    # wrapper (a caller's local, a traceback) keeps the weights alive.
    st.model = None
    st.feature_extractor = None
    st._resample_buffer = {}
    del st
    gc.collect()
    torch.cuda.empty_cache()
    freed = round((before - torch.cuda.memory_reserved()) / 2**20)
    freed_alloc = round((alloc_before - torch.cuda.memory_allocated()) / 2**20)
    print(f"[glm] speech tokenizer unloaded, VRAM reserved -{freed} MiB / allocated -{freed_alloc} MiB", flush=True)
    return freed


def _encode_prompt(name: str) -> dict:
    """Run the official prompt front-end once for profile `name`.

    Mirrors `jsonl_generate` in glmtts_inference.py: transcript normalised and
    tokenised with the trailing space (use_cache=True path), speech tokens,
    mel features for the flow, campplus speaker embedding.
    """
    if name not in _profiles:
        raise KeyError(f"unknown prompt profile {name!r}; known: {sorted(_profiles)}")
    prof = _profiles[name]
    fe, tf = _models["frontend"], _models["text_frontend"]
    if _models.get("speech_tokenizer") is None:
        t0 = time.perf_counter()
        _models["speech_tokenizer"] = _load_speech_tokenizer()
        fe.speech_tokenizer = _models["speech_tokenizer"]
        print(f"[glm] speech tokenizer reloaded for prompt encoding ({time.perf_counter() - t0:.1f}s)", flush=True)
    wav = prof["wav_abs"]
    transcript = prof["text"]
    ptext = tf.text_normalize(_t2s(transcript) if _settings["t2s_input"] else transcript)
    t0 = time.perf_counter()
    prompt_text_token = fe._extract_text_token(ptext + " ")
    prompt_speech_token = fe._extract_speech_token([wav])
    speech_feat = fe._extract_speech_feat(wav, sample_rate=SR)
    embedding = fe._extract_spk_embedding(wav)
    cst0 = prompt_speech_token.squeeze().tolist()
    enc = {
        "name": name, "wav": wav, "text": transcript, "text_norm": ptext,
        "prompt_text_token": prompt_text_token,
        "cst0": cst0,
        "flow_prompt": torch.tensor([cst0], dtype=torch.int32).to(DEVICE),
        "speech_feat": speech_feat, "embedding": embedding,
        "n_speech_tokens": len(cst0), "encode_s": round(time.perf_counter() - t0, 2),
    }
    if _settings["unload_speech_tokenizer"]:
        _drop_speech_tokenizer()
    return enc


def _apply_vram_cap() -> None:
    """v83 (RTX 3090): optional per-process VRAM cap, OFF unless GLM_VRAM_FRACTION is set.

    Found on the 24 GB card: a very short segment (「嗯？」, 14 speech tokens) made the
    flow step reserve ~37 GB -- more than the card -- which Windows (WDDM) silently backs
    with system RAM: 40 s for 0.4 s of audio, or `CUDNN_STATUS_EXECUTION_FAILED`, after
    which every request fails until restart (mainline-v83 incident note). On the 12 GB
    card that same allocation could not succeed, so the allocator fell back and nothing
    was ever seen. A cap makes an oversized allocation fail inside PyTorch, where a
    cheaper path is taken, instead of spilling. Resource only: model, precision, prompt
    and sampling are untouched."""
    frac = os.environ.get("GLM_VRAM_FRACTION", "").strip()
    if not frac or not torch.cuda.is_available():
        return
    torch.cuda.set_per_process_memory_fraction(float(frac))
    total = torch.cuda.get_device_properties(0).total_memory
    print(f"[glm] VRAM cap on: fraction {float(frac):.3f} = {round(float(frac) * total / 2**20)} MiB", flush=True)


def _load():
    global _profiles, _prompt
    if _models:
        return
    _apply_vram_cap()
    print(f"[glm] loading {ENGINE} from {REPO} (llm {LLM_DTYPE}+sdpa) ...", flush=True)
    t0 = time.perf_counter()
    _profiles = _load_profiles()
    st = _load_speech_tokenizer()
    frontend, text_frontend = load_frontends(st, sample_rate=SR, use_phoneme=False)
    llm = GLMTTS(llama_cfg_path=os.path.join("ckpt", "llm", "config.json"), mode="PRETRAIN")
    dtype = torch.float16 if LLM_DTYPE == "fp16" else torch.float32
    llm.llama = LlamaForCausalLM.from_pretrained(
        os.path.join("ckpt", "llm"), dtype=dtype, attn_implementation="sdpa").to(DEVICE)
    llm.llama_embedding = llm.llama.model.embed_tokens
    llm.set_runtime_vars(special_token_ids=get_special_token_ids(frontend.tokenize_fn))
    flow = yaml_util.load_flow_model(os.path.join("ckpt", "flow", "flow.pt"),
                                     os.path.join("ckpt", "flow", "config.yaml"), DEVICE)
    token2wav = tts_model_util.Token2Wav(flow, sample_rate=SR, device=DEVICE)
    _models.update({"frontend": frontend, "text_frontend": text_frontend, "llm": llm,
                    "token2wav": token2wav, "speech_tokenizer": st})
    # This frame must not keep its own reference: `_encode_prompt` below drops the
    # speech tokenizer through `_models`, and a live local here would keep the
    # 1.6 GB model alive (first start of v50 freed 202 MiB instead of the model).
    del st
    t_load = time.perf_counter() - t0

    _install_fast_sampler()
    _prompt = _encode_prompt(_settings["prompt"])

    # First generate pays for CUDA kernel autotuning; without it request #1 is
    # several times slower than steady state.
    t0 = time.perf_counter()
    _generate("你好，我準備好了。", _settings["seed"])
    t_warm = time.perf_counter() - t0
    print(f"[glm] load {t_load:.1f}s | prompt {_prompt['name']} encoded {_prompt['encode_s']}s "
          f"({_prompt['n_speech_tokens']} speech tokens) | warmup {t_warm:.2f}s | "
          f"unload_speech_tokenizer={_settings['unload_speech_tokenizer']} | {_vram()}", flush=True)


def _generate(synth_text: str, seed: int) -> dict:
    """One official generate_long call with a fresh cache (per-request independence,
    exactly the configuration of the v47/v48 packs the client judged)."""
    pr = _prompt
    rec = {}

    def timed_llm(**kw):
        t = time.perf_counter()
        out = local_llm_forward(**kw)
        rec["t_llm"] = rec.get("t_llm", 0.0) + (time.perf_counter() - t)
        rec["n_tokens"] = rec.get("n_tokens", 0) + len(out)
        return out

    def timed_flow(**kw):
        t = time.perf_counter()
        torch.backends.cuda.matmul.allow_tf32 = bool(_settings["tf32"])
        torch.backends.cudnn.allow_tf32 = bool(_settings["tf32"])
        steps = int(_settings["flow_steps"])
        cfg = float(_settings["flow_cfg_rate"])
        if steps == FLOW_STEPS_DEFAULT and cfg < 0:
            # untouched upstream path, so every pre-v55 number stays comparable
            out = local_flow_forward(**kw)
        else:
            t2w = kw["flow"]                      # utils.tts_model_util.Token2Wav
            prev_cfg = t2w.flow.inference_cfg_rate
            if cfg >= 0:
                t2w.flow.inference_cfg_rate = cfg
            try:
                wav, full_mel = t2w.token2wav_with_cache(
                    kw["token_list"],
                    n_timesteps=steps,
                    prompt_token=kw["prompt_speech_tokens"],
                    prompt_feat=kw["speech_feat"],
                    embedding=kw["embedding"])
                out = (wav.detach().cpu(), full_mel)
            finally:
                # never let one request's setting leak into the next one
                t2w.flow.inference_cfg_rate = prev_cfg
        rec["t_flow"] = rec.get("t_flow", 0.0) + (time.perf_counter() - t)
        rec["flow_steps"] = steps
        rec["flow_cfg_rate"] = cfg
        return out

    cache = {"cache_text": [pr["text_norm"]], "cache_text_token": [pr["prompt_text_token"]],
             "cache_speech_token": [pr["cst0"]], "use_cache": True}
    tts_speech, _, tokens, tn = generate_long(
        frontend=_models["frontend"], text_frontend=_models["text_frontend"],
        llm=_models["llm"], flow=_models["token2wav"],
        text_info=["req", synth_text], cache=cache, device=DEVICE, embedding=pr["embedding"],
        seed=seed, flow_prompt_token=pr["flow_prompt"], speech_feat=pr["speech_feat"],
        local_llm_forward=timed_llm, local_flow_forward=timed_flow)
    audio = tts_speech[0].detach().cpu().numpy().astype(np.float32)
    rec.update({"audio": audio, "segments": tn["syn_text_tn"], "n_tokens": len(tokens)})
    return rec


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
class SpeechRequest(BaseModel):
    """Subset of OpenAI's `/v1/audio/speech` body honoured here.

    `voice`, when it names a known prompt profile, selects that voice for this
    request; otherwise the configured profile is used. `model` is accepted and
    ignored (AIRI's binding still says "OmniVoice"; see /v1/models).
    """

    model: str = ENGINE
    input: str
    voice: str = ""
    response_format: str = "wav"
    speed: float = 1.0


@app.get("/v1/models")
def list_models():
    # "OmniVoice" is listed as an alias on purpose: AIRI's saved speech binding
    # names that model, and keeping the name valid is what lets this engine take
    # over with zero settings change. /health says which engine is really here.
    return {"object": "list", "data": [
        {"id": ENGINE, "object": "model", "owned_by": "local"},
        {"id": "OmniVoice", "object": "model", "owned_by": "local", "alias_of": ENGINE},
    ]}


@app.get("/health")
def health():
    # v84 F-1: "status" is now the engine's verdict ("ok" | "broken"), not a constant.
    return {"engine": ENGINE, "loaded": bool(_models),
            "prompt_loaded": _prompt.get("name"), "prompt_wav": _prompt.get("wav"),
            "speech_tokenizer_resident": _models.get("speech_tokenizer") is not None,
            "llm_dtype": LLM_DTYPE, "vram_cap_fraction": os.environ.get("GLM_VRAM_FRACTION", "") or None,
            **_counters, **_vram(), **_settings, **_engine_verdict()}


@app.get("/config")
def get_config():
    return {**_settings, "engine": ENGINE, "prompt_wav": _prompt.get("wav"),
            "prompt_text": _prompt.get("text"), "profiles": sorted(_profiles), "sample_rate": SR}


@app.post("/config")
def set_config(patch: dict):
    """Change settings without a restart. A new `prompt` re-encodes the clip
    (about two seconds, reloading the speech tokenizer if it was dropped)."""
    global _prompt, _profiles
    unknown = [k for k in patch if k not in _CONFIG_KEYS]
    if unknown:
        raise HTTPException(status_code=422, detail=f"unknown keys {unknown}; known: {list(_CONFIG_KEYS)}")
    new_prompt = patch.get("prompt")
    if new_prompt is not None and new_prompt not in _profiles:
        # profiles.json may have grown since startup (PROVENANCE.md promises "add an
        # entry, then POST /config" works without a restart) -- re-read before refusing.
        _profiles = _load_profiles()
    with _lock:
        for key in _CONFIG_KEYS:
            if key in patch and key != "prompt":
                _settings[key] = type(_settings[key])(patch[key]) if not isinstance(_settings[key], bool) \
                    else (patch[key] in (True, 1, "1", "true", "on"))
        if new_prompt is not None and new_prompt != _prompt.get("name"):
            if new_prompt not in _profiles:
                raise HTTPException(status_code=422, detail=f"unknown prompt profile {new_prompt!r}; known: {sorted(_profiles)}")
            _prompt = _encode_prompt(new_prompt)
            _settings["prompt"] = new_prompt
            print(f"[glm] prompt -> {new_prompt} ({_prompt['wav']}) encoded in {_prompt['encode_s']}s", flush=True)
        elif "unload_speech_tokenizer" in patch and _settings["unload_speech_tokenizer"]:
            _drop_speech_tokenizer()
    print(f"[glm] config -> {_settings}", flush=True)
    return {**_settings, "prompt_wav": _prompt.get("wav")}


@app.post("/debug/empty_cache")
def empty_cache():
    if not torch.cuda.is_available():
        return {"error": "no CUDA device"}
    mb = lambda n: round(n / 2**20, 1)  # noqa: E731
    before = torch.cuda.memory_reserved()
    torch.cuda.empty_cache()
    after = torch.cuda.memory_reserved()
    out = {"reserved_mb_before": mb(before), "reserved_mb_after": mb(after), "freed_mb": mb(before - after)}
    print(f"[glm] empty_cache freed {out['freed_mb']} MiB", flush=True)
    return out


@app.post("/v1/audio/speech")
def synthesize(req: SpeechRequest):
    """v84 F-1: the route counts engine failures and slow short utterances; the work is
    `_synthesize_impl`, byte-for-byte the v83 route body."""
    t0 = time.perf_counter()
    try:
        resp = _synthesize_impl(req)
    except HTTPException:
        raise                                   # the caller's bad input, not the engine
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        _engine["synth_failures"] += 1
        _engine["synth_failures_consecutive"] += 1
        _engine["last_failure"] = {"iso": datetime.now().isoformat(timespec="seconds"), "error": msg[:400],
                                   "text": (req.input or "")[:40]}
        if _CUDA_ERR.search(msg):
            _engine["cuda_error_since_start"] = True
        print(f"[glm] 🔴 SYNTHESIS FAILED ({_engine['synth_failures_consecutive']} in a row; "
              f"cuda_error_since_start={_engine['cuda_error_since_start']}): {msg[:200]} -- /health now says "
              f"{_engine_verdict()['status']}", flush=True)
        raise
    dt = time.perf_counter() - t0
    _engine["synth_ok"] += 1
    _engine["synth_failures_consecutive"] = 0
    chars = len((req.input or "").strip())
    if chars <= SLOW_SHORT_CHARS and dt > SLOW_SHORT_S:
        _engine["slow_short_utterances"] += 1
        _engine["last_slow"] = {"iso": datetime.now().isoformat(timespec="seconds"), "chars": chars, "s": round(dt, 2),
                                "text": (req.input or "")[:20], **_vram()}
        print(f"[glm] ⚠ SLOW SHORT UTTERANCE: {chars} chars took {dt:.1f}s (v83 spill symptom) {_vram()}", flush=True)
    return resp


_simulate = {"n": 0, "kind": "plain"}


@app.post("/debug/simulate_failure")
def simulate_failure(patch: dict):
    """v84 F-1 test seam: the next `n` syntheses raise (kind "plain" or "cuda") BEFORE touching
    the GPU, so the verdict can be exercised without breaking CUDA for real. n=0 cancels."""
    _simulate["n"] = int(patch.get("n", 1))
    _simulate["kind"] = str(patch.get("kind", "plain"))
    return dict(_simulate)


def _synthesize_impl(req: SpeechRequest):
    global _req_seq, _prompt
    if _simulate["n"] > 0:
        _simulate["n"] -= 1
        raise RuntimeError("CUDA error: simulated by /debug/simulate_failure" if _simulate["kind"] == "cuda"
                           else "simulated synthesis failure (/debug/simulate_failure)")

    clock = Clock()
    t_recv = clock.now()

    text = (req.input or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="input is required")

    if _settings["filter_markers"] and _looks_like_marker_debris(text):
        print(f"[glm] refused to speak marker debris: {text[:60]!r}", flush=True)
        return Response(content=_silent_wav(SR), media_type="audio/wav")

    synth_text = _t2s(text) if _settings["t2s_input"] else text
    if _settings["punct_downgrade"]:
        synth_text = _MID_PUNCT.sub("，", synth_text)

    limit = int(_settings["max_input_chars"])
    over = len(synth_text) > limit > 0
    mode = _settings["over_limit"]
    if over and mode == "error":
        raise HTTPException(status_code=413, detail=f"input is {len(synth_text)} chars, over the {limit}-char cap")
    if over and mode == "truncate":
        synth_text = synth_text[:limit]
    parts = _split_for_limit(synth_text, limit) if (over and mode == "split") else [synth_text]

    with _lock:
        _load()
        _req_seq += 1
        seq = _req_seq
        _counters["requests"] += 1
        # v55: the timing row is written AFTER the response is sent, so reading _settings
        # there attributes the row to whatever the settings happen to be by then -- a
        # /config that lands in between silently labels the row with the wrong condition.
        # Snapshot the knobs here, inside the lock, and log the snapshot (principle 10).
        _knobs = {"flow_steps": _settings["flow_steps"], "flow_cfg_rate": _settings["flow_cfg_rate"],
                  "tf32": _settings["tf32"], "fast_sampler": _settings["fast_sampler"]}

        # Per-request voice: only when the caller names a profile we know.
        prompt_switched = None
        if req.voice and req.voice in _profiles and req.voice != _prompt.get("name"):
            _prompt = _encode_prompt(req.voice)
            _settings["prompt"] = req.voice
            prompt_switched = req.voice

        tf = _models["text_frontend"]
        seed = int(_settings["seed"])
        t_synth_start = clock.now()
        chunks, t_llm, t_flow, n_tokens, segments, rerolled = [], 0.0, 0.0, 0, [], 0
        jump = None
        for part in parts:
            norm = tf.text_normalize(part)
            if not norm or not norm.strip():
                continue
            g = _generate(norm, seed)
            audio = g["audio"]
            # Voice-jump handling on the *inline* path only when reroll is on:
            # then a flagged take is regenerated once with seed+1 before we answer.
            if _settings["jump_detect"] and _settings["jump_reroll"]:
                st = onset_stats(audio, SR)
                if jump_suspect(st, _settings["jump_st_thr"], _settings["jump_db_thr"]):
                    _counters["jump_suspects"] += 1
                    print(f"[glm] #{seq} ⚠ VOICE JUMP SUSPECT d_st={st['d_st']} d_db={st['d_db']} -> reroll seed {seed + 1}", flush=True)
                    g2 = _generate(norm, seed + 1)
                    st2 = onset_stats(g2["audio"], SR)
                    rerolled += 1
                    _counters["jump_rerolls"] += 1
                    if not jump_suspect(st2, _settings["jump_st_thr"], _settings["jump_db_thr"]):
                        g, audio, st = g2, g2["audio"], st2
                    else:
                        print(f"[glm] #{seq} reroll still flagged (d_st={st2['d_st']} d_db={st2['d_db']}); keeping seed {seed}", flush=True)
                    for k in ("t_llm", "t_flow"):
                        g[k] = g.get(k, 0.0) + g2.get(k, 0.0)
                jump = st
            chunks.append(audio)
            t_llm += g.get("t_llm", 0.0)
            t_flow += g.get("t_flow", 0.0)
            n_tokens += g.get("n_tokens", 0)
            segments.extend(g.get("segments", []))
        t_synth_end = clock.now()

        if not chunks:
            print(f"[glm] #{seq} nothing to say after normalisation: {text[:40]!r}", flush=True)
            return Response(content=_silent_wav(SR), media_type="audio/wav")
        audio = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)

        head_s = head_silence_s(audio, SR, _settings["trim_head_db"])
        trimmed_s = 0.0
        if _settings["trim_head"]:
            audio, trimmed_s = trim_head(audio, SR, _settings["trim_head_db"], _settings["trim_keep_ms"])
        if _settings["normalize_loudness"]:
            audio = normalize_loudness(audio, float(_settings["target_rms"]))
        wav = _wav_bytes(audio)
        t_post = clock.now()

    audio_s = len(audio) / SR
    synth_s = t_synth_end - t_synth_start
    print(f"[glm] #{seq} {len(text)} chars -> {audio_s:.2f}s audio in {synth_s:.2f}s "
          f"(llm {t_llm:.2f} flow {t_flow:.2f}, {n_tokens} tok, RTF {synth_s / max(audio_s, 1e-3):.2f}) "
          f"head {head_s:.2f}s{' trimmed ' + format(trimmed_s, '.2f') + 's' if trimmed_s else ''} "
          f"seed={seed} prompt={_prompt['name']}", flush=True)

    audio_for_detector = audio if _settings["jump_detect"] and not _settings["jump_reroll"] else None

    def _after_send():
        # Runs after the body is handed to the transport: the detector's cost
        # (pyin, ~0.1-0.3 s) stays off the latency path unless reroll is on.
        t_sent = clock.now()
        j = jump
        if audio_for_detector is not None:
            j = onset_stats(audio_for_detector, SR)
            if not j.get("measurable"):
                _counters["jump_unmeasurable"] += 1
            elif jump_suspect(j, _settings["jump_st_thr"], _settings["jump_db_thr"]):
                _counters["jump_suspects"] += 1
                print(f"[glm] #{seq} ⚠ VOICE JUMP SUSPECT (logged, reroll off) d_st={j['d_st']} d_db={j['d_db']} "
                      f"text={text[:30]!r} seed={seed}", flush=True)
        _write_timing({
            "seq": seq, "engine": ENGINE, "chars": len(text), "text": text, "synth_text": synth_text,
            "segments": segments, "parts": len(parts), "over_limit": over,
            "prompt": _prompt["name"], "seed": seed, "prompt_switched_by_voice": prompt_switched,
            "audio_s": round(audio_s, 3), "n_tokens": n_tokens,
            "head_silence_s": round(head_s, 3), "head_trimmed_s": round(trimmed_s, 3),
            "jump": j, "jump_suspect": bool(j and jump_suspect(j, _settings["jump_st_thr"], _settings["jump_db_thr"])),
            "rerolled": rerolled,
            "wav_bytes": len(wav),
            "t_recv": t_recv, "t_synth_start": t_synth_start, "t_synth_end": t_synth_end,
            "t_post": t_post, "t_sent": t_sent,
            "iso_recv": _iso(t_recv), "iso_synth_start": _iso(t_synth_start),
            "iso_synth_end": _iso(t_synth_end), "iso_sent": _iso(t_sent),
            "queue_s": round(t_synth_start - t_recv, 4), "synth_s": round(synth_s, 4),
            "t_llm_s": round(t_llm, 4), "t_flow_s": round(t_flow, 4),
            **_knobs,
            "post_s": round(t_post - t_synth_end, 4), "write_s": round(t_sent - t_post, 4),
            "total_s": round(t_sent - t_recv, 4),
            **_vram(),
            "settings": {k: _settings[k] for k in ("trim_head", "punct_downgrade", "jump_detect", "jump_reroll",
                                                   "t2s_input", "normalize_loudness", "unload_speech_tokenizer")},
        })

    return Response(content=wav, media_type="audio/wav", background=BackgroundTask(_after_send))


async def _serve(ports: list[int]):
    servers = [uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=p, log_level="warning")) for p in ports]
    await asyncio.gather(*(s.serve() for s in servers))


if __name__ == "__main__":
    _glyph_self_test()
    port = int(os.environ.get("GLM_PORT", "5050"))
    alias = [int(p) for p in os.environ.get("GLM_ALIAS_PORTS", "").split(",") if p.strip()]
    ports = [port, *alias]
    _load()
    print(f"[glm] ready on {', '.join(f'http://127.0.0.1:{p}' for p in ports)}  (timing log: {LOG_PATH})", flush=True)
    sys.stdout.flush()
    asyncio.run(_serve(ports))
