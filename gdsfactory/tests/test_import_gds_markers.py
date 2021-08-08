from pathlib import Path

import pytest
from pytest_regressions.num_regression import NumericRegressionFixture

import gdsfactory as gf
from gdsfactory.import_gds import add_ports_from_markers_center, import_gds
from gdsfactory.port import auto_rename_ports

gdspaths = [gf.CONFIG["gdsdir"] / name for name in ["mmi1x2.gds", "mzi2x2.gds"]]


@pytest.mark.parametrize("gdspath", gdspaths)
def test_components_ports(
    gdspath: Path, num_regression: NumericRegressionFixture
) -> None:
    c = import_gds(gdspath)
    add_ports_from_markers_center(c)
    auto_rename_ports(c)
    if num_regression:
        num_regression.check(c.get_ports_array())


if __name__ == "__main__":
    c = import_gds(gdspaths[0])
    add_ports_from_markers_center(c)
    auto_rename_ports(c)
    print(c.ports.keys())
    print(c.name)

    c = import_gds(gdspaths[1])
    add_ports_from_markers_center(c)
    auto_rename_ports(c)
    print(c.ports.keys())
    print(c.name)
