"""Thin, optional bridge to Essentia.

Essentia is a heavy native dependency. Every function here degrades
gracefully: if the package is missing, :func:`essentia_available` returns
``False``, the affected stage falls back to a pure-NumPy path, and the user
sees a clear message instead of an ``ImportError`` traceback.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

#: Set to ``True`` once a successful import has happened.
_ESSENTIA: Any | None = None
_IMPORT_FAILED = False


def essentia_available() -> bool:
    """Return ``True`` if ``essentia.standard`` can be imported.

    The result is cached; Essentia's import emits a banner message, so we only
    want to pay for it once.
    """
    return get_essentia() is not None


def get_essentia() -> Any | None:
    """Import and return ``essentia.standard``, or ``None`` if unavailable.

    Returns:
        The ``essentia.standard`` module, or ``None`` when the import fails.
    """
    global _ESSENTIA, _IMPORT_FAILED
    if _ESSENTIA is not None:
        return _ESSENTIA
    if _IMPORT_FAILED:
        return None
    try:
        import essentia.standard as es  # type: ignore[import-not-found]

        _ESSENTIA = es
    except Exception as exc:  # pragma: no cover - depends on the environment
        logger.debug("essentia unavailable: %s", exc)
        _IMPORT_FAILED = True
        return None
    return _ESSENTIA


def as_essentia_signal(samples: np.ndarray) -> np.ndarray:
    """Convert a NumPy signal into the contiguous float32 array Essentia wants.

    The standard (C++) Essentia bindings in this build take mono
    ``VECTOR_REAL`` input, i.e. a flat 1-D array. A 2-D array is interpreted as
    a stereo matrix and rejected, so we pass a flat array.

    Args:
        samples: Mono float samples.

    Returns:
        A C-contiguous ``float32`` 1-D array.
    """
    return np.ascontiguousarray(np.asarray(samples, dtype=np.float32).reshape(-1))


def configure_algo(algo: Any, **params: Any) -> Any:
    """Apply parameters to an Essentia algorithm in a single call.

    Two Essentia quirks make this non-obvious:

    1. ``configure(**kwargs)`` resets every parameter not passed in the call
       back to its default. Configuring one parameter at a time therefore
       silently discards the others -- asking for ``sampleRate=22050`` and then
       ``minFrequency=65`` in separate calls leaves the algorithm at 44100 Hz
       and 80 Hz, which shows up later as a systematic octave error in the F0
       track. Everything must be passed together, once.
    2. Some parameters are declared ``INTEGER`` and reject a Python ``float``
       (``magnitudeThreshold``, ``minDuration``, ``timeContinuity``). A
       ``TypeError`` names the offending parameter, so we drop it and retry,
       leaving that one at its default.

    Args:
        algo: An Essentia standard algorithm instance.
        **params: Parameter names mapped to values. Pass integers for
            ``INTEGER`` parameters.

    Returns:
        The same algorithm instance, configured.
    """
    remaining = dict(params)
    for _attempt in range(len(params) + 1):
        if not remaining:
            break
        try:
            algo.configure(**remaining)
            return algo
        except TypeError as exc:
            dropped = _parse_bad_parameter(exc, remaining)
            if dropped is None:
                logger.debug("essentia configure failed: %s", exc)
                return algo
            logger.debug("essentia rejected %s; leaving it at its default", dropped)
            remaining.pop(dropped, None)
        except Exception:
            logger.debug("essentia build rejected parameters %s", sorted(remaining))
            return algo
    return algo


def _parse_bad_parameter(exc: TypeError, remaining: dict[str, Any]) -> str | None:
    """Extract the parameter name from an Essentia ``TypeError``, if present."""
    message = str(exc)
    for name in remaining:
        if name in message:
            return name
    return None


def require_essentia(feature: str) -> Any:
    """Return ``essentia.standard`` or raise a helpful error.

    Args:
        feature: Name of the feature being used, for the error message.

    Returns:
        The ``essentia.standard`` module.

    Raises:
        RuntimeError: If Essentia is not installed. The message names the
            install command and the fallback, because this is the first thing
            a new user hits on a plain ``pip install swaras`` and a traceback
            would be a poor way to learn it.
    """
    es = get_essentia()
    if es is None:
        raise RuntimeError(
            f"{feature} needs Essentia, which is not installed. Either:\n"
            "  pip install 'swaras[essentia]'   (Linux and macOS)\n"
            "or use the fallback:  --detector pyin\n"
            "On Windows, install Essentia inside WSL or a container."
        )
    return es
