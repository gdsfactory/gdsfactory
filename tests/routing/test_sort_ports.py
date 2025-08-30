import pytest
from kfactory import DPort

from gdsfactory.routing.sort_ports import sort_ports, sort_ports_x, sort_ports_y


def test_sort_ports_x() -> None:
    ports = [
        DPort(
            name="p1",
            width=5,
            layer=1,
            port_type="optical",
            center=(3, 0),
        ),
        DPort(
            name="p2",
            width=5,
            layer=1,
            port_type="optical",
            center=(1, 0),
        ),
        DPort(
            name="p3",
            width=5,
            layer=1,
            port_type="optical",
            center=(2, 0),
        ),
    ]
    sorted_ports = sort_ports_x(ports)
    assert [p.name for p in sorted_ports] == ["p2", "p3", "p1"]


def test_sort_ports_y() -> None:
    ports = [
        DPort(
            name="p1",
            width=0.5,
            layer=1,
            port_type="optical",
            center=(0, 3),
        ),
        DPort(
            name="p2",
            width=0.5,
            layer=1,
            port_type="optical",
            center=(0, 1),
        ),
        DPort(
            name="p3",
            width=0.5,
            layer=1,
            port_type="optical",
            center=(0, 2),
        ),
    ]
    sorted_ports = sort_ports_y(ports)
    assert [p.name for p in sorted_ports] == ["p2", "p3", "p1"]


def test_sort_ports_horizontal() -> None:
    ports1 = [
        DPort(
            name="p1",
            width=0.5,
            layer=1,
            port_type="optical",
            center=(0, 3),
        ),
        DPort(
            name="p2",
            width=0.5,
            layer=1,
            port_type="optical",
            center=(0, 1),
        ),
        DPort(
            name="p3",
            width=0.5,
            layer=1,
            port_type="optical",
            center=(0, 2),
        ),
    ]
    ports2 = [
        DPort(
            name="p4",
            width=0.5,
            layer=1,
            port_type="optical",
            orientation=180,
            center=(10, 2),
        ),
        DPort(
            name="p5",
            width=0.5,
            layer=1,
            port_type="optical",
            orientation=180,
            center=(10, 1),
        ),
        DPort(
            name="p6",
            width=0.5,
            layer=1,
            port_type="optical",
            orientation=180,
            center=(10, 3),
        ),
    ]

    sorted1, sorted2 = sort_ports(ports1, ports2, enforce_port_ordering=False)
    assert [p.name for p in sorted1] == ["p2", "p3", "p1"]
    assert [p.name for p in sorted2] == ["p5", "p4", "p6"]


def test_sort_ports_vertical() -> None:
    ports1 = [
        DPort(
            name="p1",
            width=0.5,
            layer=1,
            port_type="optical",
            orientation=90,
            center=(3, 0),
        ),
        DPort(
            name="p2",
            width=0.5,
            layer=1,
            port_type="optical",
            orientation=90,
            center=(1, 0),
        ),
        DPort(
            name="p3",
            width=0.5,
            layer=1,
            port_type="optical",
            orientation=90,
            center=(2, 0),
        ),
    ]
    ports2 = [
        DPort(
            name="p4",
            width=0.5,
            layer=1,
            port_type="optical",
            orientation=270,
            center=(2, 10),
        ),
        DPort(
            name="p5",
            width=0.5,
            layer=1,
            port_type="optical",
            orientation=270,
            center=(1, 10),
        ),
        DPort(
            name="p6",
            width=0.5,
            layer=1,
            port_type="optical",
            orientation=270,
            center=(3, 10),
        ),
    ]

    sorted1, sorted2 = sort_ports(ports1, ports2, enforce_port_ordering=False)
    assert [p.name for p in sorted1] == ["p2", "p3", "p1"]
    assert [p.name for p in sorted2] == ["p5", "p4", "p6"]


def test_sort_ports_enforce_ordering() -> None:
    ports1 = [
        DPort(
            name="p1",
            width=0.5,
            layer=1,
            port_type="optical",
            orientation=0,
            center=(0, 3),
        ),
        DPort(
            name="p2",
            width=0.5,
            layer=1,
            port_type="optical",
            orientation=0,
            center=(0, 1),
        ),
        DPort(
            name="p3",
            width=0.5,
            layer=1,
            port_type="optical",
            orientation=0,
            center=(0, 2),
        ),
    ]
    ports2 = [
        DPort(
            name="p4",
            width=0.5,
            layer=1,
            orientation=180,
            center=(10, 2),
        ),
        DPort(
            name="p5",
            width=0.5,
            layer=1,
            orientation=180,
            center=(10, 1),
        ),
        DPort(
            name="p6",
            width=0.5,
            layer=1,
            orientation=180,
            center=(10, 3),
        ),
    ]

    sorted1, sorted2 = sort_ports(ports1, ports2, enforce_port_ordering=True)
    assert [p.name for p in sorted1] == ["p2", "p3", "p1"]
    assert [p.name for p in sorted2] == ["p5", "p6", "p4"]


def test_sort_ports_mixed_orientation() -> None:
    ports1 = [
        DPort(
            name="p1",
            width=0.5,
            layer=1,
            port_type="optical",
            orientation=0,
            center=(0, 3),
        ),
        DPort(
            name="p2",
            width=0.5,
            layer=1,
            port_type="optical",
            orientation=0,
            center=(0, 1),
        ),
    ]
    ports2 = [
        DPort(
            name="p3",
            width=0.5,
            layer=1,
            port_type="optical",
            orientation=90,
            center=(10, 2),
        ),
        DPort(
            name="p4",
            width=0.5,
            layer=1,
            port_type="optical",
            orientation=90,
            center=(10, 1),
        ),
    ]

    sorted1, sorted2 = sort_ports(ports1, ports2, enforce_port_ordering=False)
    assert [p.name for p in sorted1] == ["p2", "p1"]
    assert [p.name for p in sorted2] == ["p4", "p3"]

    sorted1, sorted2 = sort_ports(ports1, ports2, enforce_port_ordering=True)
    assert [p.name for p in sorted1] == ["p2", "p1"]
    assert [p.name for p in sorted2] == ["p4", "p3"]


def test_sort_ports_validation() -> None:
    ports1 = [
        DPort(
            name="p1",
            width=0.5,
            layer=1,
            port_type="optical",
            orientation=0,
            center=(0, 0),
        )
    ]
    ports2 = [
        DPort(
            name="p2",
            width=0.5,
            layer=1,
            port_type="optical",
            orientation=180,
            center=(10, 0),
        ),
        DPort(
            name="p3",
            width=0.5,
            layer=1,
            port_type="optical",
            orientation=180,
            center=(10, 1),
        ),
    ]

    with pytest.raises(ValueError, match="ports1=1 and ports2=2 must be equal"):
        sort_ports(ports1, ports2, enforce_port_ordering=False)

    with pytest.raises(ValueError, match="ports1 is an empty list"):
        sort_ports([], [], enforce_port_ordering=False)
