from pathlib import Path

import pytest
from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf
from gdsfactory.add_ports import add_ports_from_markers_center

# gdspaths = [gf.CONFIG["gdsdir"] / name for name in ["mmi1x2.gds", "mzi2x2.gds"]]
gdspaths = [gf.CONFIG["gdsdir"] / name for name in ["mzi2x2.gds"]]


@pytest.mark.parametrize("gdspath", gdspaths)
def test_components_ports(
    gdspath: Path, data_regression: DataRegressionFixture
) -> None:
    """Read ports from markers."""
    c = gf.import_gds(gdspath, decorator=add_ports_from_markers_center)
    data_regression.check(c.to_dict())


if __name__ == "__main__":
    c = gf.import_gds(gdspaths[0], decorator=add_ports_from_markers_center)
    # c = c.copy()
    # add_ports_from_markers_center(c)
    # c.auto_rename_ports()
    # print(c.ports.keys())
    # print(c.name)
    d = c.to_dict()
    c.show(show_ports=True)
