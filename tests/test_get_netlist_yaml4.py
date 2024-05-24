from __future__ import annotations

import gdsfactory as gf
from gdsfactory.config import PATH


def test_netlist_read_translated_rotated() -> None:
    filepath = PATH.netlists / "bend_translated_rotated.yml"
    c = gf.read.from_yaml(filepath)
    assert len(c.insts) == 1, len(c.insts)


def regenerate_regression_test() -> None:
    c = gf.Component()
    ref = c.add_ref(gf.components.bend_circular())
    ref.name = "b1"
    ref = c.add_ref(gf.components.bend_circular())
    ref.name = "b2"
    ref.d.movex(10)
    ref.mirror()
    ref.d.rotate(180)
    filepath = PATH.netlists / "bend_translated_rotated.yml"
    c.name = "original"
    c.write_netlist(filepath)


if __name__ == "__main__":
    regenerate_regression_test()
    gf.clear_cache()
    filepath = PATH.netlists / "bend_translated_rotated.yml"
    c = gf.read.from_yaml(filepath)
    c.name = "reconstructed"
