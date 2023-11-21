import pytest

import gdsfactory as gf
from gdsfactory.port import PortOrientationError


@gf.cell
def non_manhattan_component() -> gf.Component:
    p = gf.path.arc(radius=10, angle=45)
    s0 = gf.Section(width=1, offset=3, layer=(2, 0), name="waveguide")
    s1 = gf.Section(
        width=1, offset=0, layer=(1, 0), name="heater", port_names=("o1", "o2")
    )
    xs = gf.CrossSection(sections=(s0, s1))
    return gf.path.extrude(p, xs)


def test_non_manhattan_warn() -> None:
    with pytest.warns(UserWarning):
        assert non_manhattan_component()


@pytest.mark.skip("TODO: fix this test")
def test_non_manhattan_error() -> None:
    """TODO: fix this test."""
    default = gf.CONF.ports_not_manhattan
    gf.CONF.ports_not_manhattan = "error"
    with pytest.raises(PortOrientationError):
        assert non_manhattan_component()
    gf.CONF.ports_not_manhattan = default


if __name__ == "__main__":
    # test_non_manhattan_warn()
    # test_non_manhattan_error()

    default = gf.CONF.ports_not_manhattan
    gf.CONF.ports_not_manhattan = "error"

    c = non_manhattan_component()

    # with pytest.raises(PortOrientationError):
    #     assert non_manhattan_component()
    # gf.CONF.ports_not_manhattan = default
