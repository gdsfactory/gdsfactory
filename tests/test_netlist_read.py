from __future__ import annotations

import gdsfactory as gf
from gdsfactory.config import PATH


def test_netlist_read() -> None:
    filepath = PATH.netlists / "mzi.yml"
    c = gf.read.from_yaml(filepath)
    assert len(c.get_dependencies()) == 5, len(c.get_dependencies())


def regenerate_regression_test() -> None:
    c = gf.components.mzi()
    filepath = PATH.netlists / "mzi.yml"
    c.write_netlist(filepath)


if __name__ == "__main__":
    test_netlist_read()
