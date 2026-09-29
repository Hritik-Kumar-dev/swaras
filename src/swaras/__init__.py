"""Audio-to-Swar: transcribe a melodic recording as Indian swar notation.

Pipeline stages
---------------
``audio_io`` -> ``pitch`` -> ``tonic`` -> ``normalize`` -> ``segment``
-> ``swar`` / ``shruti`` -> ``format``

Each stage is an independent module with small, testable functions.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
