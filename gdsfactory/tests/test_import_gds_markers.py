from pathlib import Path

import pytest
from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf
from gdsfactory.add_ports import add_ports_from_markers_center
from gdsfactory.port import auto_rename_ports

# gdspaths = [gf.CONFIG["gdsdir"] / name for name in ["mmi1x2.gds", "mzi2x2.gds"]]
gdspaths = [gf.CONFIG["gdsdir"] / name for name in ["mzi2x2.gds"]]


@pytest.mark.parametrize("gdspath", gdspaths)
def test_components_ports(
    gdspath: Path, data_regression: DataRegressionFixture
) -> gf.Component:
    c = gf.import_gds(gdspath)
    add_ports_from_markers_center(c)
    auto_rename_ports(c)
    data_regression.check(c.to_dict())


if __name__ == "__main__":
    c = gf.import_gds(gdspaths[0])
    add_ports_from_markers_center(c)
    auto_rename_ports(c)
    print(c.ports.keys())
    print(c.name)
