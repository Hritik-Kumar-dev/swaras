"""Tests for the shruti / swara tuning table.

The table is data, so these tests guard the data and the loader rather than an
algorithm. The musicological assumptions are documented in
``docs/shruti_table.md``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from swaras.tuning import ShrutiTable, load_shruti_table

#: The 22-shruti just-intonation values the task specifies.
EXPECTED_SHRUTI_CENTS = [
    0, 90, 112, 182, 204, 294, 316, 386, 408, 498, 519,
    590, 612, 702, 792, 814, 884, 906, 996, 1018, 1088, 1110,
]


@pytest.fixture(scope="module")
def table() -> ShrutiTable:
    """The default tuning table."""
    return load_shruti_table()


def test_table_has_22_shrutis(table: ShrutiTable) -> None:
    assert table.n_shrutis == 22


def test_table_has_12_swaras(table: ShrutiTable) -> None:
    assert table.n_swaras == 12


def test_shruti_cents_match_the_specified_table(table: ShrutiTable) -> None:
    assert table.shruti_cents.tolist() == EXPECTED_SHRUTI_CENTS


def test_shruti_cents_strictly_increasing(table: ShrutiTable) -> None:
    cents = table.shruti_cents
    assert all(b > a for a, b in zip(cents, cents[1:]))


def test_all_shrutis_within_one_octave(table: ShrutiTable) -> None:
    assert all(0 <= c < 1200 for c in table.shruti_cents)


def test_swar_positions_in_cents_ascending(table: ShrutiTable) -> None:
    cents = [s.cents for s in table.swaras]
    assert all(b > a for a, b in zip(cents, cents[1:]))


def test_swar_names_are_the_twelve_expected(table: ShrutiTable) -> None:
    assert [s.name for s in table.swaras] == [
        "Sa", "Re komal", "Re", "Ga komal", "Ga", "Ma",
        "Ma tivra", "Pa", "Dha komal", "Dha", "Ni komal", "Ni",
    ]


@pytest.mark.parametrize(
    "swar,expected",
    [
        ("Sa", [0]),
        ("Re komal", [90, 112]),
        ("Re", [182, 204]),
        ("Ga komal", [294, 316]),
        ("Ga", [386, 408]),
        ("Ma", [498, 519]),
        ("Ma tivra", [590, 612]),
        ("Pa", [702]),
        ("Dha komal", [792, 814]),
        ("Dha", [884, 906]),
        ("Ni komal", [996, 1018]),
        ("Ni", [1088, 1110]),
    ],
)
def test_shruti_to_swar_mapping(table: ShrutiTable, swar: str, expected: list[int]) -> None:
    """Each swara owns the shruti(s) documented in docs/shruti_table.md."""
    assert [s.cents for s in table.shrutis_for(swar)] == expected


def test_every_shruti_belongs_to_a_known_swar(table: ShrutiTable) -> None:
    names = {s.name for s in table.swaras}
    assert all(s.swar in names for s in table.shrutis)


def test_every_shruti_is_covered_exactly_once(table: ShrutiTable) -> None:
    seen = [s.cents for swar in table.swaras for s in table.shrutis_for(swar.name)]
    assert sorted(seen) == sorted(EXPECTED_SHRUTI_CENTS)


def test_nominal_cents_are_pair_means(table: ShrutiTable) -> None:
    """Each swara's nominal cents must be the mean of its shruti pair."""
    for swar in table.swaras:
        members = table.shrutis_for(swar.name)
        if len(members) == 2:
            assert swar.cents == pytest.approx((members[0].cents + members[1].cents) / 2, abs=0.5)
        else:  # a singleton such as Sa or Pa is its own nominal
            assert swar.cents == pytest.approx(members[0].cents, abs=0.5)


def test_sa_is_zero_cents(table: ShrutiTable) -> None:
    assert table.swaras[0].name == "Sa"
    assert table.swaras[0].cents == 0


def test_pa_is_702_cents(table: ShrutiTable) -> None:
    pa = table.swar_by_name("Pa")
    assert pa.cents == 702


def test_swar_lookup_by_name(table: ShrutiTable) -> None:
    assert table.swar_by_name("Ni").cents == 1099
    with pytest.raises(KeyError):
        table.swar_by_name("Ti")


def test_table_repr_is_informative(table: ShrutiTable) -> None:
    assert "22" in repr(table)
    assert "12" in repr(table)


def test_loader_accepts_explicit_path() -> None:
    from swaras.config import DEFAULT_SHRUTI_TABLE

    t = load_shruti_table(DEFAULT_SHRUTI_TABLE)
    assert t.n_shrutis == 22


def test_loader_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_shruti_table(Path("/nonexistent/table.json"))


@pytest.fixture
def base_table_json() -> dict:
    """A mutable copy of the packaged table, for negative tests to corrupt."""
    from swaras.config import DEFAULT_SHRUTI_TABLE

    return json.loads(Path(DEFAULT_SHRUTI_TABLE).read_text(encoding="utf-8"))


def test_loader_rejects_short_shruti_list(tmp_path: Path, base_table_json: dict) -> None:
    bad = tmp_path / "bad.json"
    base_table_json["shrutis"] = base_table_json["shrutis"][:5]
    bad.write_text(json.dumps(base_table_json))
    with pytest.raises(ValueError, match="22 shrutis"):
        load_shruti_table(bad)


def test_loader_rejects_unsorted_shruti_cents(tmp_path: Path, base_table_json: dict) -> None:
    bad = tmp_path / "bad.json"
    s = base_table_json["shrutis"]
    s[3], s[4] = s[4], s[3]
    bad.write_text(json.dumps(base_table_json))
    with pytest.raises(ValueError, match="increasing"):
        load_shruti_table(bad)


def test_loader_rejects_unknown_swar_reference(tmp_path: Path, base_table_json: dict) -> None:
    bad = tmp_path / "bad.json"
    base_table_json["shrutis"][5]["swar"] = "Ti"
    bad.write_text(json.dumps(base_table_json))
    with pytest.raises(ValueError, match="unknown swara"):
        load_shruti_table(bad)


def test_loader_rejects_wrong_swara_count(tmp_path: Path, base_table_json: dict) -> None:
    bad = tmp_path / "bad.json"
    base_table_json["swar_positions"] = base_table_json["swar_positions"][:11]
    bad.write_text(json.dumps(base_table_json))
    with pytest.raises(ValueError, match="12 swara positions"):
        load_shruti_table(bad)


def test_loader_rejects_out_of_octave_shruti(tmp_path: Path, base_table_json: dict) -> None:
    bad = tmp_path / "bad.json"
    base_table_json["shrutis"][-1]["cents"] = 1300
    bad.write_text(json.dumps(base_table_json))
    with pytest.raises(ValueError, match="within one octave"):
        load_shruti_table(bad)


def test_loader_rejects_malformed_json(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_shruti_table(bad)


def test_loader_rejects_missing_key(tmp_path: Path, base_table_json: dict) -> None:
    bad = tmp_path / "bad.json"
    del base_table_json["shrutis"]
    bad.write_text(json.dumps(base_table_json))
    with pytest.raises(ValueError, match="missing required key"):
        load_shruti_table(bad)
