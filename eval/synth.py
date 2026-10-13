"""W3-2 — render the eval utterances to audio. docs/12 §2.

Three voices, chosen to differ in accent and register. The plan (docs/00 D5)
names Cartesia; with no API key in this environment the local macOS `say`
engine is used instead, which keeps synthesis free, offline and exactly
reproducible. The caveat in D5 applies with more force, not less: `say` is
cleaner and more regular than any human, so absolute accuracy is inflated.
The comparison that matters — clean vs noise vs phone — is still sound.

Output: eval/dataset/audio/<case>__<voice>__clean.wav, 16 kHz mono PCM16,
plus audio_manifest.json (committed; the wav files are gitignored and
regenerate from it).

Run: .venv/bin/python -m eval.synth
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import soundfile as sf

from eval.dataset import DATASET, load_cases

AUDIO = DATASET / "audio"
SAMPLE_RATE = 16_000
VOICES = {
    "v1": {"engine": "say", "voice": "Samantha", "note": "US English, female"},
    "v2": {"engine": "say", "voice": "Daniel", "note": "UK English, male"},
    "v3": {"engine": "say", "voice": "Rishi", "note": "Indian English, male"},
}


def render(text: str, voice: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["say", "-v", voice, "-o", str(path), f"--data-format=LEI16@{SAMPLE_RATE}", text],
        check=True,
    )


def main() -> int:
    cases = load_cases()
    manifest = {"sample_rate": SAMPLE_RATE, "voices": VOICES, "files": []}
    for case in cases:
        for vid, voice in VOICES.items():
            path = AUDIO / f"{case.id}__{vid}__clean.wav"
            if not path.exists():
                render(case.utterance, voice["voice"], path)
            info = sf.info(str(path))
            manifest["files"].append({
                "case_id": case.id, "voice": vid, "variant": "clean",
                "path": str(path.relative_to(DATASET)),
                "text_sha": hashlib.sha256(case.utterance.encode()).hexdigest()[:12],
                "seconds": round(info.duration, 3),
            })
    (DATASET / "audio_manifest.json").write_text(json.dumps(manifest, indent=1))
    total = sum(f["seconds"] for f in manifest["files"])
    print(f"synth: {len(manifest['files'])} files, {total / 60:.1f} minutes of audio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
