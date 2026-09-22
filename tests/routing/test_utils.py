import pytest

import gdsfactory as gf
from gdsfactory.routing.utils import (
    BendPortTypeError,
    check_ports_have_equal_spacing,
    direction_ports_from_list_ports,
    get_default_bend,
    get_list_ports_angle,
    validate_bend90,
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


@pytest.mark.parametrize(
    ("port_type", "cross_section", "expected"),
    [
        ("electrical", "metal_routing", "wire_corner"),
        ("electrical", "gs", "wire_corner_sections"),
        ("optical", "strip", "bend_euler"),
        # Optical, even though it has an electrical heater section.
        ("optical", "strip_heater_metal", "bend_euler"),
        ("vertical_te", "strip", "bend_euler"),
    ],
)
def test_get_default_bend(port_type: str, cross_section: str, expected: str) -> None:
    xs = gf.get_cross_section(cross_section)
    assert get_default_bend(port_type, xs) == expected


def test_validate_bend90() -> None:
    validate_bend90(
        gf.get_component("bend_euler", cross_section="strip"), "optical", "bend_euler"
    )
    validate_bend90(
        gf.get_component("wire_corner", cross_section="metal_routing"),
        "electrical",
        "wire_corner",
    )


def test_validate_bend90_wrong_port_type() -> None:
    corner = gf.get_component("wire_corner", cross_section="strip")
    with pytest.raises(BendPortTypeError, match="0 'optical' ports") as excinfo:
        validate_bend90(corner, "optical", "bend_euler")
    assert "Use bend='bend_euler'" in str(excinfo.value)
    assert "['electrical', 'electrical']" in str(excinfo.value)


def test_validate_bend90_default_is_not_suggested() -> None:
    # bend_euler takes the port types of the metal cross-section it is drawn with.
    bend = gf.get_component("bend_euler", cross_section="metal_routing")
    with pytest.raises(BendPortTypeError, match="0 'optical' ports") as excinfo:
        validate_bend90(bend, "optical", "bend_euler")
    assert "Use bend=" not in str(excinfo.value)
