import pytest

from gdsfactory.routing.utils import (
    check_ports_have_equal_spacing,
    direction_ports_from_list_ports,
    get_list_ports_angle,
)
from gdsfactory.typings import Port


def test_direction_ports_from_list_ports() -> None:
    ports = [
        Port(
            name="p1",
            width=0.5,
            layer=1,
            port_type="optical",
            orientation=0,
            center=(0, 0),
        ),
        Port(
            name="p2",
            width=0.5,
            layer=1,
            port_type="optical",
            orientation=90,
            center=(0, 0),
        ),
        Port(
            name="p3",
            width=0.5,
            layer=1,
            port_type="optical",
            orientation=180,
            center=(0, 0),
        ),
        Port(
            name="p4",
            width=0.5,
            layer=1,
            port_type="optical",
            orientation=270,
            center=(0, 0),
        ),
        Port(
            name="p5",
            width=0.5,
            layer=1,
            port_type="optical",
            orientation=0,
            center=(0, 1),
        ),
        Port(
            name="p6",
            width=0.5,
            layer=1,
            port_type="optical",
            orientation=90,
            center=(1, 0),
        ),
    ]

    result = direction_ports_from_list_ports(ports)

    assert len(result["E"]) == 2
    assert len(result["N"]) == 2
    assert len(result["W"]) == 1
    assert len(result["S"]) == 1

    assert result["E"][0].dy < result["E"][1].dy
    assert result["N"][0].dx < result["N"][1].dx


def test_check_ports_have_equal_spacing() -> None:
    ports_h = [
        Port(
            name="p1",
            width=0.5,
            layer=1,
            port_type="optical",
            orientation=0,
            center=(0, 0),
        ),
        Port(
            name="p2",
            width=0.5,
            layer=1,
            port_type="optical",
            orientation=0,
            center=(0, 1),
        ),
        Port(
            name="p3",
            width=0.5,
            layer=1,
            port_type="optical",
            orientation=0,
            center=(0, 2),
        ),
    ]
    assert check_ports_have_equal_spacing(ports_h) == 1.0

    ports_v = [
        Port(
            name="p1",
            width=0.5,
            layer=1,
            port_type="optical",
            orientation=90,
            center=(0, 0),
        ),
        Port(
            name="p2",
            width=0.5,
            layer=1,
            port_type="optical",
            orientation=90,
            center=(1, 0),
        ),
        Port(
            name="p3",
            width=0.5,
            layer=1,
            port_type="optical",
            orientation=90,
            center=(2, 0),
        ),
    ]
    assert check_ports_have_equal_spacing(ports_v) == 1.0

    ports_unequal = [
        Port(
            name="p1",
            width=0.5,
            layer=1,
            port_type="optical",
            orientation=0,
            center=(0, 0),
        ),
        Port(
            name="p2",
            width=0.5,
            layer=1,
            port_type="optical",
            orientation=0,
            center=(0, 1),
        ),
        Port(
            name="p3",
            width=0.5,
            layer=1,
            port_type="optical",
            orientation=0,
            center=(0, 2.5),
        ),
    ]
    with pytest.raises(ValueError, match="Ports should have the same separation"):
        check_ports_have_equal_spacing(ports_unequal)

    with pytest.raises(ValueError, match="list_ports should be a list of ports"):
        check_ports_have_equal_spacing(tuple())

    with pytest.raises(ValueError, match="list_ports should not be empty"):
        check_ports_have_equal_spacing([])


def test_get_list_ports_angle() -> None:
    ports_single = [
        Port(
            name="p1",
            width=0.5,
            layer=1,
            port_type="optical",
            orientation=90,
            center=(0, 0),
        )
    ]
    assert get_list_ports_angle(ports_single) == 90

    ports_same = [
        Port(
            name="p1",
            width=0.5,
            layer=1,
            port_type="optical",
            orientation=90,
            center=(0, 0),
        ),
        Port(
            name="p2",
            width=0.5,
            layer=1,
            port_type="optical",
            orientation=90,
            center=(1, 0),
        ),
    ]
    assert get_list_ports_angle(ports_same) == 90

    assert get_list_ports_angle([]) is None

    ports_different = [
        Port(
            name="p1",
            width=0.5,
            layer=1,
            port_type="optical",
            orientation=90,
            center=(0, 0),
        ),
        Port(
            name="p2",
            width=0.5,
            layer=1,
            port_type="optical",
            orientation=0,
            center=(1, 0),
        ),
    ]
    with pytest.raises(ValueError, match="All port angles should be the same"):
        get_list_ports_angle(ports_different)
