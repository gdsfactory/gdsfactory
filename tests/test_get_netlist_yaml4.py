from __future__ import annotations

import gdsfactory as gf
from gdsfactory.config import PATH


def test_netlist_read_translated_rotated() -> None:
    filepath = PATH.netlists / "bend_translated_rotated.yml"
    c = gf.read.from_yaml(filepath)
    assert len(c.insts) == 2, len(c.insts)


def regenerate_regression_test() -> None:
    c = gf.Component(name="test_netlist_yaml4")
    ref = c.add_ref(gf.components.bend_circular())
    ref.name = "b1"
    ref = c.add_ref(gf.components.bend_circular())
    ref.name = "b2"
    ref.dmovex(10)
    ref.dmirror()
    ref.rotate(180)
    filepath = PATH.netlists / "bend_translated_rotated.yml"
    n = c.get_netlist()
    c.write_netlist(n, filepath)


if __name__ == "__main__":
    regenerate_regression_test()
    gf.clear_cache()
    filepath = PATH.netlists / "bend_translated_rotated.yml"
    c = gf.read.from_yaml(filepath)
