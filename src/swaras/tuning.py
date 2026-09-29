"""Loading and validation of the swara / shruti tuning table.

The tuning lives in a JSON config file (see ``config/shruti_table.json`` and
``docs/shruti_table.md``) so that the musicology is data rather than code.
This module turns that file into typed, validated objects.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import DEFAULT_SHRUTI_TABLE

#: The number of shrutis and swara positions the default table must define.
EXPECTED_N_SHRUTIS = 22
EXPECTED_N_SWARAS = 12


@dataclass(frozen=True)
class Swara:
    """One of the 12 swara positions within an octave.

    Attributes:
        name: Human-readable name, e.g. ``"Re komal"``.
        short: Compact symbol for text output, e.g. ``"r"``.
        cents: Nominal position above Sa, in cents.
        alternate: Alternative name, if any.
    """

    name: str
    short: str
    cents: float
    alternate: str | None = None


@dataclass(frozen=True)
class Shruti:
    """One of the 22 shrutis within an octave.

    Attributes:
        index: Position in the table, ``0`` to ``21``.
        cents: Position above Sa, in cents.
        swar: Name of the swara this shruti belongs to.
        variant: ``"low"``/``"high"`` within a pair, or ``"shuddha"`` for a
            singleton such as Sa or Pa.
    """

    index: int
    cents: float
    swar: str
    variant: str = "shuddha"


@dataclass(frozen=True)
class ShrutiTable:
    """A validated tuning table.

    Attributes:
        name: Table name, for display.
        swaras: The 12 swara positions, ordered by pitch.
        shrutis: The 22 shrutis, ordered by pitch.
    """

    name: str
    swaras: tuple[Swara, ...]
    shrutis: tuple[Shruti, ...]

    @property
    def n_swaras(self) -> int:
        """Number of swara positions (12)."""
        return len(self.swaras)

    @property
    def n_shrutis(self) -> int:
        """Number of shrutis (22)."""
        return len(self.shrutis)

    @property
    def swara_cents(self) -> np.ndarray:
        """Nominal cents of the 12 swara positions."""
        return np.array([s.cents for s in self.swaras], dtype=np.float64)

    @property
    def shruti_cents(self) -> np.ndarray:
        """Cents of the 22 shrutis."""
        return np.array([s.cents for s in self.shrutis], dtype=np.float64)

    def swar_by_name(self, name: str) -> Swara:
        """Look up a swara position by name.

        Args:
            name: Swara name, e.g. ``"Re komal"``.

        Returns:
            The matching :class:`Swara`.

        Raises:
            KeyError: If no swara has that name.
        """
        for swar in self.swaras:
            if swar.name == name:
                return swar
        raise KeyError(f"unknown swara {name!r}")

    def shrutis_for(self, swar_name: str) -> list[Shruti]:
        """Return the shrutis belonging to one swara, in pitch order.

        Args:
            swar_name: Swara name.

        Returns:
            List of one shruti, or a 22-cent pair for an altered swara.

        Raises:
            KeyError: If no swara has that name.
        """
        self.swar_by_name(swar_name)  # validates the name
        return [s for s in self.shrutis if s.swar == swar_name]

    def __repr__(self) -> str:
        return f"ShrutiTable(name={self.name!r}, shrutis={self.n_shrutis}, swaras={self.n_swaras})"


def _parse_swaras(raw: list[dict[str, Any]]) -> tuple[Swara, ...]:
    """Build :class:`Swara` objects from raw JSON.

    Args:
        raw: The ``swar_positions`` list.

    Returns:
        Tuple of swara objects ordered by cents.

    Raises:
        ValueError: If the count is wrong or the list is not ascending.
    """
    if len(raw) != EXPECTED_N_SWARAS:
        raise ValueError(f"expected {EXPECTED_N_SWARAS} swara positions, got {len(raw)}")
    swaras = tuple(
        Swara(name=str(r["name"]), short=str(r["short"]), cents=float(r["cents"]), alternate=r.get("alternate"))
        for r in raw
    )
    cents = [s.cents for s in swaras]
    if any(b <= a for a, b in zip(cents, cents[1:])):
        raise ValueError("swara positions must be in strictly increasing cents order")
    return swaras


def _parse_shrutis(raw: list[dict[str, Any]], known: set[str]) -> tuple[Shruti, ...]:
    """Build :class:`Shruti` objects from raw JSON and validate them.

    Args:
        raw: The ``shrutis`` list.
        known: Names of the swara positions, used to check references.

    Returns:
        Tuple of shruti objects ordered by cents.

    Raises:
        ValueError: If the count is wrong, the list is not ascending, or a
            shruti references an unknown swara.
    """
    if len(raw) != EXPECTED_N_SHRUTIS:
        raise ValueError(f"expected {EXPECTED_N_SHRUTIS} shrutis, got {len(raw)}")
    shrutis = tuple(
        Shruti(
            index=int(r["index"]),
            cents=float(r["cents"]),
            swar=str(r["swar"]),
            variant=str(r.get("variant", "shuddha")),
        )
        for r in raw
    )
    cents = [s.cents for s in shrutis]
    if any(b <= a for a, b in zip(cents, cents[1:])):
        raise ValueError("shrutis must be in strictly increasing cents order")
    if any(s.cents < 0 or s.cents >= 1200 for s in shrutis):
        raise ValueError("shrutis must lie within one octave (0 <= cents < 1200)")
    unknown = {s.swar for s in shrutis} - known
    if unknown:
        raise ValueError(f"shrutis reference unknown swara(s): {sorted(unknown)}")
    return shrutis


def load_shruti_table(path: str | Path | None = None) -> ShrutiTable:
    """Load and validate a tuning table from JSON.

    Args:
        path: Path to the table; defaults to the packaged
            :data:`~swaras.config.DEFAULT_SHRUTI_TABLE`.

    Returns:
        The validated table.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file is malformed, or fails validation: wrong
            number of shrutis or swaras, non-ascending cents, out-of-octave
            values, or a shruti naming an unknown swara.
    """
    table_path = Path(path) if path is not None else DEFAULT_SHRUTI_TABLE
    if not table_path.is_file():
        raise FileNotFoundError(f"shruti table not found: {table_path}")
    try:
        data = json.loads(table_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {table_path}: {exc}") from exc
    try:
        swaras = _parse_swaras(data["swar_positions"])
        shrutis = _parse_shrutis(data["shrutis"], {s.name for s in swaras})
    except KeyError as exc:
        raise ValueError(f"{table_path} is missing required key {exc}") from exc
    return ShrutiTable(name=str(data.get("name", table_path.stem)), swaras=swaras, shrutis=shrutis)
