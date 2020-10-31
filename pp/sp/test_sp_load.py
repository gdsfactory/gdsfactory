import pytest
import pp
from pp.components import component_factory

component_types = [
    "waveguide",
    "bend_circular",
    # "bend_euler90",
    "coupler",
    "mmi1x2",
    "mmi2x2",
]


@pytest.mark.parametrize("component_type", component_types)
def test_sp_load(component_type, data_regression):
    c = component_factory[component_type]()
    sp = pp.sp.load(c)

    port_names = sp[0]
    f = list(sp[1])
    s = sp[2]

    lenf = s.shape[0]
    rows = s.shape[1]
    cols = s.shape[2]

    assert rows == cols == len(c.ports)
    assert len(port_names) == len(c.ports)
    data_regression.check(dict(port_names=port_names))
    assert lenf == len(f)


if __name__ == "__main__":
    c = pp.c.waveguide(layer=(2, 0))
    print(c.get_sparameters_path())
