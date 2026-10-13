"""W3-3 — the 8 kHz mu-law phone variant. docs/12 §4.

    16 kHz PCM16 -> bandpass 300-3400 Hz -> 8 kHz -> mu-law encode -> decode
                 -> back to 16 kHz

The final upsample is deliberate: every STT provider then receives the same
format, so the measured difference is the CODEC DAMAGE, not an input-format
mismatch between providers.

Applied to the clean and snr10 variants.

Run: .venv/bin/python -m eval.telephony   (after eval.noise_mix)
"""

from __future__ import annotations

import json
import sys

import numpy as np
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfiltfilt

from eval.dataset import DATASET

SAMPLE_RATE = 16_000
MU = 255.0
SOURCES = {"clean": "telephony_clean", "snr10": "telephony_snr10"}


def mu_law_round_trip(x: np.ndarray) -> np.ndarray:
    """G.711 mu-law: compand, quantize to 8 bits, expand."""
    x = np.clip(x, -1.0, 1.0)
    companded = np.sign(x) * np.log1p(MU * np.abs(x)) / np.log1p(MU)
    quantized = np.round((companded + 1) / 2 * 255) / 255 * 2 - 1
    return np.sign(quantized) * ((1 + MU) ** np.abs(quantized) - 1) / MU


def phone(audio: np.ndarray) -> np.ndarray:
    sos = butter(4, [300, 3400], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    band = sosfiltfilt(sos, audio)
    narrow = resample_poly(band, 1, 2)            # 16 kHz -> 8 kHz
    coded = mu_law_round_trip(narrow)
    wide = resample_poly(coded, 2, 1)             # back to 16 kHz
    return np.clip(wide[: len(audio)], -1, 1).astype(np.float32)


def main() -> int:
    path = DATASET / "audio_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"] = [f for f in manifest["files"] if not f["variant"].startswith("telephony")]
    new = []
    for entry in manifest["files"]:
        if entry["variant"] not in SOURCES:
            continue
        audio, _ = sf.read(str(DATASET / entry["path"]), dtype="float32")
        variant = SOURCES[entry["variant"]]
        out = DATASET / "audio" / f"{entry['case_id']}__{entry['voice']}__{variant}.wav"
        sf.write(str(out), phone(audio), SAMPLE_RATE, subtype="PCM_16")
        new.append({**entry, "variant": variant, "path": str(out.relative_to(DATASET))})
    manifest["files"].extend(new)
    path.write_text(json.dumps(manifest, indent=1))
    print(f"telephony: {len(new)} files · manifest now {len(manifest['files'])} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
