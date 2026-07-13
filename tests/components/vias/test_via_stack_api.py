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


@pytest.mark.xfail(
    strict=True,
    reason="DeprecationWarning not yet emitted; will be wired up in Task 8",
)
def test_legacy_positional_call_still_works_with_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        c = via_stack((5.0, 5.0), ("M1", "M2"), None, ("via1", None))
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    d = c.to_dict(with_ports=True)
    assert d["settings"]["size"] == (5.0, 5.0)
    assert d["settings"]["layers"] == ("M1", "M2")
