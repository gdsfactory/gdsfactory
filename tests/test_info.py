"""Calculated/derived properties are stored in info."""

from __future__ import annotations

import gdsfactory as gf


def test_args() -> None:
    c1 = gf.components.pad((150, 150))
    assert c1.info["xsize"] == 150
