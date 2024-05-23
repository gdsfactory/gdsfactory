from __future__ import annotations

import gdsfactory as gf


def test_write_cells():
    c = gf.c.mzi()
    assert len(c.called_cells()) == 8
    assert c.child_cells() == 8
