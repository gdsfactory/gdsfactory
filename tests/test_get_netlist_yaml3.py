from __future__ import annotations

import gdsfactory as gf
from gdsfactory.config import PATH


def test_netlist_read_rotated() -> None:
    filepath = PATH.netlists / "bend_rotated.yml"
    c = gf.read.from_yaml(filepath)
    assert len(c.insts) == 1, len(c.insts)


def regenerate_regression_test() -> None:
    c = gf.Component("test_get_netlist_yaml3")
    ref = c.add_ref(gf.components.bend_circular())
    ref.drotate(30)
    filepath = PATH.netlists / "bend_rotated.yml"
    c.write_netlist(filepath)


if __name__ == "__main__":
    regenerate_regression_test()
    gf.clear_cache()
    filepath = PATH.netlists / "bend_rotated.yml"
    c = gf.read.from_yaml(filepath)
    c.show()
