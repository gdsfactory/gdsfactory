###################################################################################################################
# PROPRIETARY AND CONFIDENTIAL
# THIS SOFTWARE IS THE SOLE PROPERTY AND COPYRIGHT (c) 2021 OF ROCKLEY PHOTONICS LTD.
# USE OR REPRODUCTION IN PART OR AS A WHOLE WITHOUT THE WRITTEN AGREEMENT OF ROCKLEY PHOTONICS LTD IS PROHIBITED.
# RPLTD NOTICE VERSION: 1.1.1
###################################################################################################################
import numpy as np
import pytest

import gdsfactory as gf

DEMO_PORT_ANGLE = 10


@pytest.fixture
def shear_waveguide_symmetric():
    P = gf.path.straight(length=10)
    c = gf.path.extrude(
        P, "strip", shear_angle_start=DEMO_PORT_ANGLE, shear_angle_end=DEMO_PORT_ANGLE
    )
    return c


@pytest.fixture
def shear_waveguide_start():
    P = gf.path.straight(length=10)
    c = gf.path.extrude(
        P, "strip", shear_angle_start=DEMO_PORT_ANGLE, shear_angle_end=None
    )
    return c


@pytest.fixture
def shear_waveguide_end():
    P = gf.path.straight(length=10)
    c = gf.path.extrude(
        P, "strip", shear_angle_start=None, shear_angle_end=DEMO_PORT_ANGLE
    )
    return c


@pytest.fixture
def regular_waveguide():
    P = gf.path.straight(length=10)
    c = gf.path.extrude(P, "strip")
    return c


def test_mate_on_shear_xor_empty(
    regular_waveguide, shear_waveguide_start, shear_waveguide_end
):
    # two sheared components joined at the sheared port should appear the same as two straight component joined
    two_straights = gf.Component()
    c1 = two_straights << regular_waveguide
    c2 = two_straights << regular_waveguide
    c2.connect("o1", c1.ports["o2"])

    two_shears = gf.Component()
    c1 = two_shears << shear_waveguide_end
    c2 = two_shears << shear_waveguide_start
    c2.connect("o1", c1.ports["o2"])

    xor = gf.geometry.xor_diff(two_straights, two_shears)
    assert not xor.layers


def test_area_stays_same(
    regular_waveguide,
    shear_waveguide_start,
    shear_waveguide_end,
    shear_waveguide_symmetric,
):
    components = [
        regular_waveguide,
        shear_waveguide_start,
        shear_waveguide_end,
        shear_waveguide_symmetric,
    ]
    areas = [c.area() for c in components]
    np.testing.assert_allclose(areas, desired=areas[0])


def test_shear_angle_annotated_on_ports(shear_waveguide_start, shear_waveguide_end):
    assert shear_waveguide_start.ports["o1"].shear_angle == DEMO_PORT_ANGLE
    assert shear_waveguide_start.ports["o2"].shear_angle is None

    assert shear_waveguide_end.ports["o2"].shear_angle == DEMO_PORT_ANGLE
    assert shear_waveguide_end.ports["o1"].shear_angle is None


@pytest.mark.skip
def test_port_attributes(regular_waveguide, shear_waveguide_symmetric):
    # TODO: this test should pass... why does it not? it seems to be ignoring the information supplied to the port originally and overwriting with info extracted from the face segment, which is not what it should do in this case. It should have the same attributes of an orthogonal port, but only the shear_angle attribute is different
    regular_ports = [p.to_dict() for p in regular_waveguide.ports.values()]
    shear_ports = [p.to_dict() for p in shear_waveguide_symmetric.ports.values()]

    for p in shear_ports:
        shear_angle = p.pop("shear_angle")
        assert shear_angle == DEMO_PORT_ANGLE

    assert regular_ports[0] == shear_ports[0]
    assert regular_ports[1] == shear_ports[1]
