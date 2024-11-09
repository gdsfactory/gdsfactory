from __future__ import annotations

import gdsfactory as gf
from gdsfactory.config import PATH


def test_netlist_read() -> None:
    filepath = PATH.netlists / "mzi.yml"
    c = gf.read.from_yaml(filepath)
    assert len(c.insts) == 16, len(c.insts)


def regenerate_regression_test() -> None:
    c = gf.components.mzi(delta_length=0.123)
    filepath = PATH.netlists / "mzi.yml"
    n = c.to_yaml()
    c.write_netlist(n, filepath)


if __name__ == "__main__":
    regenerate_regression_test()
    gf.clear_cache()
    filepath = PATH.netlists / "mzi.yml"
    c = gf.read.from_yaml(filepath)
    c.show()
