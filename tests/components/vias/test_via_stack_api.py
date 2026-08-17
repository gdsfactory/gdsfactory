from __future__ import annotations

import warnings

import pytest

from gdsfactory.components.vias.via_stack import via_stack


def test_default_call_settings_snapshot() -> None:
    c = via_stack()
    d = c.to_dict(with_ports=True)
    assert d["settings"]["size"] == (11.0, 11.0)
    assert d["settings"]["layers"] == ("M1", "M2", "MTOP")
    assert d["settings"]["vias"] == ("via1", "via2", None)


def test_legacy_positional_call_still_works_with_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        c = via_stack((5.0, 5.0), ("M1", "M2"), None, ("via1", None))
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert c.info["xsize"] == 5.0
    assert c.info["ysize"] == 5.0


def test_bottom_top_layer_convenience() -> None:
    c = via_stack(bottom_layer="M1", top_layer="M2", via_between="via1")
    d = c.to_dict(with_ports=True)
    # The cell decorator captures call-time args; check bottom_layer/top_layer directly.
    assert d["settings"]["bottom_layer"] == "M1"
    assert d["settings"]["top_layer"] == "M2"
    assert d["settings"]["via_between"] == "via1"


def test_bottom_top_layer_rejects_mixing_with_layers() -> None:
    # Passing a non-default layers= together with bottom_layer/top_layer must raise.
    with pytest.raises(ValueError):
        via_stack(bottom_layer="M1", top_layer="M2", layers=("M1", "M2"))


def test_columns_rows_cap_via_count() -> None:
    c_uncapped = via_stack(size=(20.0, 20.0))
    c_capped = via_stack(size=(20.0, 20.0), columns=1, rows=1)

    # r.na is the kfactory array column-count for the via instance (0 when columns=1).
    def sum_via_na(c: object) -> int:
        return sum(r.na for r in c.insts if "via" in r.cell.name.lower())  # type: ignore[union-attr]

    assert sum_via_na(c_capped) < sum_via_na(c_uncapped)
