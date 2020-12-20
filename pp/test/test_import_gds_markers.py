import pytest

import pp
from pp.import_gds import add_ports_from_markers_center, import_gds
from pp.port import auto_rename_ports

gdspaths = [pp.CONFIG["gdsdir"] / name for name in ["mmi1x2.gds", "mzi2x2.gds"]]


@pytest.mark.parametrize("gdspath", gdspaths)
def test_components_ports(gdspath, num_regression):
    c = import_gds(gdspath)
    add_ports_from_markers_center(c)
    auto_rename_ports(c)
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
