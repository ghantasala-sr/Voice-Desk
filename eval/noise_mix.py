"""W3-3 — mix noise into the clean eval audio at fixed SNRs. docs/12 §3.

THE DETAIL THAT MAKES THE NUMBERS CORRECT: SNR is computed over ACTIVE SPEECH
FRAMES ONLY. A file that is 40% silence has its true speech-to-noise ratio
misstated by several dB if signal power is averaged across the silence, and
every number in the table quietly moves with it.

Noise sources, all generated locally (no downloads):

  babble  six voices NOT used in the eval set, reading corpus sentences at
          once — the plan's own recipe, and the hardest case for a call
          centre because it is speech-shaped
  pink    synthetic 1/f noise. A broadband stand-in, NOT a café recording
  brown   synthetic 1/f² noise. A low-frequency rumble stand-in, NOT traffic

Real café and street recordings would be better. They are a download, and so
are left for when that has been agreed.

Run: .venv/bin/python -m eval.noise_mix
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from eval.dataset import DATASET, load_cases

SAMPLE_RATE = 16_000
SNRS = {"snr20": 20.0, "snr10": 10.0, "snr5": 5.0}
NOISE_TYPES = ["babble", "pink", "brown"]
NOISE_DIR = DATASET / "noise"
AUDIO = DATASET / "audio"
FRAME = 320  # 20 ms
SEED = 20261013

BABBLE_VOICES = ["Karen", "Moira", "Tessa", "Fred", "Albert", "Kathy"]
BABBLE_TEXT = [
    "Mobile tickets are released forty eight hours before doors open at most venues.",
    "The garage opens two hours before doors and fills quickly on sold out nights.",
    "Transfers must be accepted before the event start time or they expire.",
    "Service fees are refunded in full whenever the ticket price is refunded.",
    "Bag check is available inside the main entrance until thirty minutes after the show.",
    "Listings stay up until the event starts, and you can remove them at any time.",
]


# ---------------------------------------------------------------------------
# Noise generation
# ---------------------------------------------------------------------------
def colored_noise(kind: str, seconds: float, rng: np.random.Generator) -> np.ndarray:
    """Pink (1/f) or brown (1/f^2) noise by spectral shaping of white noise."""
    n = int(seconds * SAMPLE_RATE)
    spectrum = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, 1 / SAMPLE_RATE)
    freqs[0] = freqs[1]
    exponent = {"pink": 0.5, "brown": 1.0}[kind]
    shaped = np.fft.irfft(spectrum / freqs**exponent, n)
    return (shaped / np.max(np.abs(shaped))).astype(np.float32)


def babble(seconds: float) -> np.ndarray:
    tracks = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, voice in enumerate(BABBLE_VOICES):
            path = Path(tmp) / f"{i}.wav"
            text = " ".join(BABBLE_TEXT[i:] + BABBLE_TEXT[:i]) * 3
            subprocess.run(["say", "-v", voice, "-o", str(path),
                            f"--data-format=LEI16@{SAMPLE_RATE}", text], check=True)
            audio, _ = sf.read(str(path), dtype="float32")
            reps = int(np.ceil(seconds * SAMPLE_RATE / len(audio)))
            tracks.append(np.tile(audio, reps)[: int(seconds * SAMPLE_RATE)])
    mixed = np.sum(tracks, axis=0)
    return (mixed / np.max(np.abs(mixed))).astype(np.float32)


def ensure_noise(seconds: float = 90.0) -> dict[str, np.ndarray]:
    NOISE_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    out = {}
    for kind in NOISE_TYPES:
        path = NOISE_DIR / f"{kind}.wav"
        if not path.exists():
            audio = babble(seconds) if kind == "babble" else colored_noise(kind, seconds, rng)
            sf.write(str(path), audio, SAMPLE_RATE, subtype="PCM_16")
        out[kind], _ = sf.read(str(path), dtype="float32")
    return out


# ---------------------------------------------------------------------------
# Mixing
# ---------------------------------------------------------------------------
def active_mask(speech: np.ndarray, *, floor_db: float = -35.0) -> np.ndarray:
    """Per-sample mask of frames carrying speech.

    Energy-based: a frame is active if its RMS is within `floor_db` of the
    loudest frame. Crude next to a trained VAD, but deterministic, and on
    synthesized speech with true digital silence between words it is exact
    enough — the residual SNR error is checked and reported (see measure_snr).
    """
    frames = len(speech) // FRAME
    rms = np.sqrt(np.mean(speech[: frames * FRAME].reshape(frames, FRAME) ** 2, axis=1) + 1e-12)
    active = 20 * np.log10(rms / rms.max()) > floor_db
    mask = np.repeat(active, FRAME)
    return np.pad(mask, (0, len(speech) - len(mask)), constant_values=False)


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float,
               rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (mixed, scaled_noise, gain). Noise is a random, seeded slice."""
    offset = int(rng.integers(0, len(noise) - len(speech)))
    segment = noise[offset: offset + len(speech)]
    mask = active_mask(speech)
    p_speech = np.mean(speech[mask] ** 2)
    p_noise = np.mean(segment**2)
    scaled = segment * np.sqrt(p_speech / (p_noise * 10 ** (snr_db / 10)))
    mixed = speech + scaled
    # Prevent clipping by scaling BOTH together, which leaves the SNR intact.
    peak = np.max(np.abs(mixed))
    gain = 0.97 / peak if peak > 0.97 else 1.0
    return (mixed * gain).astype(np.float32), (scaled * gain).astype(np.float32), gain


def measure_snr(speech: np.ndarray, noise: np.ndarray) -> float:
    """Independent re-measurement of what was actually mixed."""
    mask = active_mask(speech)
    return float(10 * np.log10(np.mean(speech[mask] ** 2) / np.mean(noise**2)))


def main() -> int:
    noises = ensure_noise()
    rng = np.random.default_rng(SEED)
    manifest_path = DATASET / "audio_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"] = [f for f in manifest["files"] if f["variant"] == "clean"]
    clean = list(manifest["files"])
    case_index = {c.id: i for i, c in enumerate(load_cases())}

    errors = []
    for entry in clean:
        speech, _ = sf.read(str(DATASET / entry["path"]), dtype="float32")
        # Noise type cycles by case so every voice meets every noise type.
        kind = NOISE_TYPES[case_index[entry["case_id"]] % len(NOISE_TYPES)]
        for variant, snr in SNRS.items():
            mixed, noise_part, gain = mix_at_snr(speech, noises[kind], snr, rng)
            path = AUDIO / f"{entry['case_id']}__{entry['voice']}__{variant}.wav"
            sf.write(str(path), mixed, SAMPLE_RATE, subtype="PCM_16")
            # Independent check: speech and noise at the SAME gain the mix used.
            measured = measure_snr(speech * gain, noise_part)
            errors.append(abs(measured - snr))
            manifest["files"].append({**entry, "variant": variant, "noise": kind,
                                      "snr_db": snr, "snr_measured_db": round(measured, 2),
                                      "path": str(path.relative_to(DATASET))})
    manifest_path.write_text(json.dumps(manifest, indent=1))
    print(f"noise_mix: {len(clean) * len(SNRS)} files · SNR error max {max(errors):.3f} dB "
          f"mean {np.mean(errors):.3f} dB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
