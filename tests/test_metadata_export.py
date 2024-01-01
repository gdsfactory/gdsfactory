from __future__ import annotations

from functools import partial

import toolz

import gdsfactory as gf


def test_metadata_export_partial() -> None:
    straight_wide = partial(gf.components.straight, width=2)
    c = straight_wide()
    d = c.to_dict()
    assert d["settings"]["width"] == 2


def test_metadata_export_compose() -> None:
    straight_wide = toolz.compose(gf.components.extend_ports, gf.components.straight)
    c = straight_wide()
    d = c.to_dict()
    assert d["function_name"] == "extend_ports"
    assert d["settings"]["length"] == 5
