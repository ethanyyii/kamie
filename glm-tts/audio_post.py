"""Audio post-processing shared by the GLM adapter and its offline tools. numpy/librosa only.

Kept out of the adapter so the tools (jump-detector calibration, provenance
checks) can import these functions without importing torch or loading models.

Three things live here, each answering one of the v49 decision-file gaps:

  head silence   GLM emits 0.2-0.8 s of silence before the first phone (the tight
                 prompt r09 brought the median to ~0.22 s, v48). The browser's
                 `audio_start` mark fires when playback begins, so this silence is
                 invisible to the end-to-end timeline but audible as a delay. It is
                 measured on every take and, when `trim_head` is on, cut.

  voice jump     ~1/12 takes start in a different voice for the first syllable or
                 two (v48: same seed reproduces it; the signature is an onset that
                 is BOTH lower in pitch and quieter than the rest of the take --
                 dF0 -1.77 st / dRMS -2.09 dB on the known case, while ordinary
                 takes open +2..+6 st / +1..+8 dB). `onset_stats` measures that
                 pair; `jump_suspect` applies the thresholds. Calibrated against
                 the v48 seed batch by `tools/jump_calibrate_v50.py` -- see
                 `jump_calibration_v50.json` for the recorded margins.

  loudness       linear gain toward a target RMS with a hard peak cap (no
                 clipping, no waveshaping). Same rule as the OmniVoice adapter.
"""

import math

import numpy as np


def _frame_rms(x: np.ndarray, sr: int, frame_ms: float = 10.0) -> np.ndarray:
    n = max(1, int(sr * frame_ms / 1000))
    if len(x) < n:
        return np.array([float(np.sqrt((x ** 2).mean()))]) if len(x) else np.zeros(1)
    m = len(x) // n
    frames = x[: m * n].reshape(m, n)
    return np.sqrt((frames ** 2).mean(axis=1))


def _db(v: float) -> float:
    return 20.0 * math.log10(max(v, 1e-9))


def head_silence_s(x: np.ndarray, sr: int, thr_db: float = -40.0, frame_ms: float = 10.0) -> float:
    """Seconds before the first 10 ms frame whose RMS clears `thr_db` dBFS.

    Absolute threshold, not relative to the take's peak: a take that is quiet
    throughout should not have its whole opening declared silence.
    """
    rms = _frame_rms(x, sr, frame_ms)
    thr = 10 ** (thr_db / 20)
    idx = np.argmax(rms > thr) if (rms > thr).any() else len(rms)
    return float(idx * frame_ms / 1000)


def trim_head(x: np.ndarray, sr: int, thr_db: float = -40.0, keep_ms: float = 40.0,
              fade_ms: float = 8.0) -> tuple[np.ndarray, float]:
    """Cut leading silence, keeping `keep_ms` of it so the onset is not abrupt,
    with a short fade-in so the cut can never click. Returns (audio, seconds removed)."""
    hs = head_silence_s(x, sr, thr_db)
    cut_s = max(0.0, hs - keep_ms / 1000)
    n = int(cut_s * sr)
    if n <= 0:
        return x, 0.0
    y = x[n:].copy()
    f = min(len(y), int(sr * fade_ms / 1000))
    if f > 0:
        y[:f] *= np.linspace(0.0, 1.0, f, dtype=y.dtype)
    return y, cut_s


def onset_stats(x: np.ndarray, sr: int, onset_s: float = 0.35, min_rest_s: float = 0.5,
                fmin: float = 100.0, fmax: float = 600.0) -> dict:
    """Pitch and level of the first `onset_s` of speech versus the rest.

    Expects audio whose head silence is already trimmed (or measured: the onset
    window is taken from the first frame above the silence threshold).
    Returns None-valued fields when either side has too little voiced audio to
    say anything -- the caller must treat that as "not measurable", not "fine".
    """
    import librosa

    start = int(head_silence_s(x, sr) * sr)
    y = x[start:]
    n_on = int(onset_s * sr)
    on, rest = y[:n_on], y[n_on:]
    out = {"onset_s": onset_s, "rest_s": round(len(rest) / sr, 3),
           "f0_onset": None, "f0_rest": None, "d_st": None,
           "rms_onset_db": None, "rms_rest_db": None, "d_db": None, "measurable": False}
    if len(rest) < min_rest_s * sr or len(on) < n_on:
        return out

    hop = 256
    frame = 1024

    def voiced_medians(seg):
        f0, vflag, _ = librosa.pyin(seg.astype(np.float32), fmin=fmin, fmax=fmax, sr=sr,
                                    frame_length=frame, hop_length=hop)
        rms = librosa.feature.rms(y=seg.astype(np.float32), frame_length=frame, hop_length=hop)[0]
        m = min(len(f0), len(rms))
        f0, vflag, rms = f0[:m], vflag[:m], rms[:m]
        v = vflag & np.isfinite(f0)
        if v.sum() < 3:
            return None, None, int(v.sum())
        return float(np.median(f0[v])), float(np.median(rms[v])), int(v.sum())

    f_on, r_on, n_von = voiced_medians(on)
    f_re, r_re, n_vre = voiced_medians(rest)
    out.update({"voiced_frames_onset": n_von, "voiced_frames_rest": n_vre})
    if f_on is None or f_re is None:
        return out
    out.update({
        "f0_onset": round(f_on, 1), "f0_rest": round(f_re, 1),
        "d_st": round(12 * math.log2(f_on / f_re), 2),
        "rms_onset_db": round(_db(r_on), 2), "rms_rest_db": round(_db(r_re), 2),
        "d_db": round(_db(r_on) - _db(r_re), 2),
        "measurable": True,
    })
    return out


def jump_suspect(stats: dict, st_thr: float = -1.0, db_thr: float = -1.5) -> bool:
    """The v48 signature: onset lower in pitch AND quieter than the rest.
    Either alone is ordinary phrase-initial variation; the conjunction is not."""
    if not stats.get("measurable"):
        return False
    return stats["d_st"] <= st_thr and stats["d_db"] <= db_thr


def normalize_loudness(x: np.ndarray, target_rms: float, peak_cap: float = 0.95) -> np.ndarray:
    rms = float(np.sqrt((x ** 2).mean())) if len(x) else 0.0
    peak = float(np.abs(x).max()) if len(x) else 0.0
    if rms <= 1e-4 or peak <= 0:
        return x
    gain = min(target_rms / rms, peak_cap / peak)
    return x * gain
