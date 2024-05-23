from __future__ import annotations

import gdsfactory as gf
from gdsfactory.config import PATH


def test_netlist_read_translated_rotated() -> None:
    filepath = PATH.netlists / "bend_mirror.yml"
    c = gf.read.from_yaml(filepath)
    assert len(c.insts) == 1, len(c.insts)


def regenerate_regression_test() -> None:
    c = gf.Component()
    b1 = c.add_ref(gf.components.bend_circular())
    b1.name = "b1"
    b2 = c.add_ref(gf.components.bend_circular())
    b2.name = "b2"
    b2.d.mirror_x(10)
    filepath = PATH.netlists / "bend_mirror.yml"
    c.name = "original"
    c.write_netlist(filepath, connection_error_types={})
    c.show()


if __name__ == "__main__":
    regenerate_regression_test()
    gf.clear_cache()
    filepath = PATH.netlists / "bend_mirror.yml"
    c = gf.read.from_yaml(filepath)
    c.name = "reconstructed"
    c.show()
