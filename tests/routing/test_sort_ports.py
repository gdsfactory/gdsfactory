import pytest
from kfactory.kcell import Port

from gdsfactory.routing.sort_ports import sort_ports, sort_ports_x, sort_ports_y


def test_sort_ports_x() -> None:
    ports = [
        Port(
            name="p1",
            dwidth=0.5,
            layer=1,
            port_type="optical",
            dangle=0,
            dcenter=(3, 0),
        ),
        Port(
            name="p2",
            dwidth=0.5,
            layer=1,
            port_type="optical",
            dangle=0,
            dcenter=(1, 0),
        ),
        Port(
            name="p3",
            dwidth=0.5,
            layer=1,
            port_type="optical",
            dangle=0,
            dcenter=(2, 0),
        ),
    ]
    sorted_ports = sort_ports_x(ports)
    assert [p.name for p in sorted_ports] == ["p2", "p3", "p1"]


def test_sort_ports_y() -> None:
    ports = [
        Port(
            name="p1",
            dwidth=0.5,
            layer=1,
            port_type="optical",
            dangle=0,
            dcenter=(0, 3),
        ),
        Port(
            name="p2",
            dwidth=0.5,
            layer=1,
            port_type="optical",
            dangle=0,
            dcenter=(0, 1),
        ),
        Port(
            name="p3",
            dwidth=0.5,
            layer=1,
            port_type="optical",
            dangle=0,
            dcenter=(0, 2),
        ),
    ]
    sorted_ports = sort_ports_y(ports)
    assert [p.name for p in sorted_ports] == ["p2", "p3", "p1"]


def test_sort_ports_horizontal() -> None:
    ports1 = [
        Port(
            name="p1",
            dwidth=0.5,
            layer=1,
            port_type="optical",
            dangle=0,
            dcenter=(0, 3),
        ),
        Port(
            name="p2",
            dwidth=0.5,
            layer=1,
            port_type="optical",
            dangle=0,
            dcenter=(0, 1),
        ),
        Port(
            name="p3",
            dwidth=0.5,
            layer=1,
            port_type="optical",
            dangle=0,
            dcenter=(0, 2),
        ),
    ]
    ports2 = [
        Port(
            name="p4",
            dwidth=0.5,
            layer=1,
            port_type="optical",
            dangle=180,
            dcenter=(10, 2),
        ),
        Port(
            name="p5",
            dwidth=0.5,
            layer=1,
            port_type="optical",
            dangle=180,
            dcenter=(10, 1),
        ),
        Port(
            name="p6",
            dwidth=0.5,
            layer=1,
            port_type="optical",
            dangle=180,
            dcenter=(10, 3),
        ),
    ]

    sorted1, sorted2 = sort_ports(ports1, ports2, enforce_port_ordering=False)
    assert [p.name for p in sorted1] == ["p2", "p3", "p1"]
    assert [p.name for p in sorted2] == ["p5", "p4", "p6"]


def test_sort_ports_vertical() -> None:
    ports1 = [
        Port(
            name="p1",
            dwidth=0.5,
            layer=1,
            port_type="optical",
            dangle=90,
            dcenter=(3, 0),
        ),
        Port(
            name="p2",
            dwidth=0.5,
            layer=1,
            port_type="optical",
            dangle=90,
            dcenter=(1, 0),
        ),
        Port(
            name="p3",
            dwidth=0.5,
            layer=1,
            port_type="optical",
            dangle=90,
            dcenter=(2, 0),
        ),
    ]
    ports2 = [
        Port(
            name="p4",
            dwidth=0.5,
            layer=1,
            port_type="optical",
            dangle=270,
            dcenter=(2, 10),
        ),
        Port(
            name="p5",
            dwidth=0.5,
            layer=1,
            port_type="optical",
            dangle=270,
            dcenter=(1, 10),
        ),
        Port(
            name="p6",
            dwidth=0.5,
            layer=1,
            port_type="optical",
            dangle=270,
            dcenter=(3, 10),
        ),
    ]

    sorted1, sorted2 = sort_ports(ports1, ports2, enforce_port_ordering=False)
    assert [p.name for p in sorted1] == ["p2", "p3", "p1"]
    assert [p.name for p in sorted2] == ["p5", "p4", "p6"]


def test_sort_ports_enforce_ordering() -> None:
    ports1 = [
        Port(
            name="p1",
            dwidth=0.5,
            layer=1,
            port_type="optical",
            dangle=0,
            dcenter=(0, 3),
        ),
        Port(
            name="p2",
            dwidth=0.5,
            layer=1,
            port_type="optical",
            dangle=0,
            dcenter=(0, 1),
        ),
        Port(
            name="p3",
            dwidth=0.5,
            layer=1,
            port_type="optical",
            dangle=0,
            dcenter=(0, 2),
        ),
    ]
    ports2 = [
        Port(
            name="p4",
            dwidth=0.5,
            layer=1,
            port_type="optical",
            dangle=180,
            dcenter=(10, 2),
        ),
        Port(
            name="p5",
            dwidth=0.5,
            layer=1,
            port_type="optical",
            dangle=180,
            dcenter=(10, 1),
        ),
        Port(
            name="p6",
            dwidth=0.5,
            layer=1,
            port_type="optical",
            dangle=180,
            dcenter=(10, 3),
        ),
    ]

    sorted1, sorted2 = sort_ports(ports1, ports2, enforce_port_ordering=True)
    assert [p.name for p in sorted1] == ["p2", "p3", "p1"]
    assert [p.name for p in sorted2] == ["p5", "p6", "p4"]


def test_sort_ports_mixed_orientation() -> None:
    ports1 = [
        Port(
            name="p1",
            dwidth=0.5,
            layer=1,
            port_type="optical",
            dangle=0,
            dcenter=(0, 3),
        ),
        Port(
            name="p2",
            dwidth=0.5,
            layer=1,
            port_type="optical",
            dangle=0,
            dcenter=(0, 1),
        ),
    ]
    ports2 = [
        Port(
            name="p3",
            dwidth=0.5,
            layer=1,
            port_type="optical",
            dangle=90,
            dcenter=(10, 2),
        ),
        Port(
            name="p4",
            dwidth=0.5,
            layer=1,
            port_type="optical",
            dangle=90,
            dcenter=(10, 1),
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
        Port(
            name="p1",
            dwidth=0.5,
            layer=1,
            port_type="optical",
            dangle=0,
            dcenter=(0, 0),
        )
    ]
    ports2 = [
        Port(
            name="p2",
            dwidth=0.5,
            layer=1,
            port_type="optical",
            dangle=180,
            dcenter=(10, 0),
        ),
        Port(
            name="p3",
            dwidth=0.5,
            layer=1,
            port_type="optical",
            dangle=180,
            dcenter=(10, 1),
        ),
    ]

    with pytest.raises(ValueError, match="ports1=1 and ports2=2 must be equal"):
        sort_ports(ports1, ports2, enforce_port_ordering=False)

    with pytest.raises(ValueError, match="ports1 is an empty list"):
        sort_ports([], [], enforce_port_ordering=False)


if __name__ == "__main__":
    test_sort_ports_validation()
    test_sort_ports_horizontal()
    test_sort_ports_vertical()
    test_sort_ports_mixed_orientation()
    test_sort_ports_enforce_ordering()
    test_sort_ports_x()
    test_sort_ports_y()
