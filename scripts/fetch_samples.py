"""Regenerate the audio files under ``samples/``.

The samples are deliberately not committed: three of them are LibriSpeech
excerpts fetched over the network, and shipping third-party audio in the
repository is both bulky and a licensing question. The two synthetic files are
generated, and the three recordings are downloaded, so both can be rebuilt from
scratch with::

    python scripts/fetch_samples.py

Nothing in the test suite depends on these files. They are only used by the
accuracy report scripts, which skip any sample that is missing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"
SR = 22_050

# The generators live in tests/, which is not an installed package, so the
# repository root has to go on the path explicitly.
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

#: LibriSpeech excerpts, keyed by the name ``librosa.example`` uses. These are
#: spoken English, fetched read-only and used only as a real-voice test signal.
#: They have no musical tonic, so they are useful for pitch-tracking comparison
#: and nothing more.
LIBRI = ["libri1", "libri2", "libri3"]


def write(name: str, y: np.ndarray, sample_rate: int = SR) -> None:
    """Write a sample to ``samples/``.

    Args:
        name: Filename stem.
        y: Samples to write.
        sample_rate: Sample rate in Hz.
    """
    SAMPLES.mkdir(exist_ok=True)
    path = SAMPLES / f"{name}.wav"
    sf.write(path, np.asarray(y, dtype=np.float32), sample_rate)
    print(f"wrote {path.relative_to(ROOT)}")


def make_sweep() -> None:
    """A logarithmic 80 -> 900 Hz sweep, for checking F0 tracking."""
    from tests import synthetic

    write("sweep", synthetic.sine_sweep(80.0, 900.0, 6.0, SR))


def make_phrase() -> None:
    """Sa-Re-Ga-Pa-Dha-Ni twice at 261.63 Hz, the known ground truth."""
    from tests import synthetic

    y, _ = synthetic.melodic_phrase(261.63, repeat=2, note_duration_s=0.45)
    write("phrase_c4", y)


def fetch_libri() -> None:
    """Download the LibriSpeech excerpts via librosa's example registry."""
    try:
        import librosa
    except ImportError:
        print("skipping the LibriSpeech samples: librosa is not installed", file=sys.stderr)
        return
    for key in LIBRI:
        try:
            path = librosa.example(key)
            y, sr = librosa.load(path, sr=None, mono=True)
            write(key, y, sr)
        except Exception as exc:  # network, or the registry moved
            print(f"could not fetch {key}: {exc}", file=sys.stderr)


def main() -> int:
    """Build every sample.

    Returns:
        Process exit code.
    """
    make_sweep()
    make_phrase()
    fetch_libri()
    print(f"\n{len(list(SAMPLES.glob('*.wav')))} files in {SAMPLES.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
