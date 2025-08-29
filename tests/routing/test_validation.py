import pytest

import gdsfactory as gf
from gdsfactory.port import Port
from gdsfactory.routing.utils import RouteWarning
from gdsfactory.routing.validation import is_invalid_bundle_topology, make_error_traces


def test_make_error_traces() -> None:
    c = gf.Component()
    ports1 = [
        Port(name="p1", orientation=0, center=(0, 0), width=0.5, layer=1),
        Port(name="p2", orientation=0, center=(0, 10), width=0.5, layer=1),
    ]
    ports2 = [
        Port(name="p3", orientation=180, center=(10, 0), width=0.5, layer=1),
        Port(name="p4", orientation=180, center=(10, 10), width=0.5, layer=1),
    ]

    with pytest.warns(RouteWarning, match="test message"):
        make_error_traces(c, ports1, ports2, "test message")

    assert len(c.insts) == 2


def test_is_invalid_bundle_topology() -> None:
    p1 = [Port(name="p1", orientation=90, center=(0, 0), width=0.5, layer=1)]
    p2 = [Port(name="p2", orientation=180, center=(10, 0), width=0.5, layer=1)]
    assert not is_invalid_bundle_topology(p1, p2)

    p1_no_angle = Port(name="p1", orientation=0, center=(0, 0), width=0.5, layer=1)
    p2_no_angle = Port(name="p2", orientation=0, center=(10, 0), width=0.5, layer=1)
    assert not is_invalid_bundle_topology([p1_no_angle], [p2_no_angle])

    p1 = [
        Port(name="p1", orientation=90, center=(0, 0), width=0.5, layer=1),
        Port(name="p2", orientation=90, center=(0, 10), width=0.5, layer=1),
    ]
    p2 = [
        Port(name="p3", orientation=180, center=(10, 0), width=0.5, layer=1),
        Port(name="p4", orientation=180, center=(10, 10), width=0.5, layer=1),
    ]
    assert not is_invalid_bundle_topology(p1, p2)

    p1 = [
        Port(name="p1", orientation=45, center=(0, 0), width=0.5, layer=1),
        Port(name="p2", orientation=-45, center=(0, 10), width=0.5, layer=1),
    ]
    p2 = [
        Port(name="p3", orientation=-135, center=(10, 10), width=0.5, layer=1),
        Port(name="p4", orientation=135, center=(10, 0), width=0.5, layer=1),
    ]
    assert is_invalid_bundle_topology(p1, p2)

    p1 = [
        Port(name="p1", orientation=-135, center=(0, 0), width=0.5, layer=1),
        Port(name="p2", orientation=135, center=(0, 10), width=0.5, layer=1),
    ]
    p2 = [
        Port(name="p3", orientation=45, center=(10, 10), width=0.5, layer=1),
        Port(name="p4", orientation=-45, center=(10, 0), width=0.5, layer=1),
    ]
    assert is_invalid_bundle_topology(p1, p2)

    p1 = [
        Port(name="p1", orientation=90, center=(0, 0), width=0.5, layer=1),
        Port(name="p2", orientation=90, center=(0, 10), width=0.5, layer=1),
    ]
    p2 = [
        Port(name="p3", orientation=90, center=(10, 5), width=0.5, layer=1),
        Port(name="p4", orientation=90, center=(10, 15), width=0.5, layer=1),
    ]
    assert is_invalid_bundle_topology(p1, p2)

    p1 = [
        Port(name="p1", orientation=90, center=(0, 0), width=0.5, layer=1),
        Port(name="p2", orientation=90, center=(0, 10), width=0.5, layer=1),
        Port(name="p3", orientation=180, center=(0, 20), width=0.5, layer=1),
    ]
    p2 = [
        Port(name="p4", orientation=180, center=(10, 0), width=0.5, layer=1),
        Port(name="p5", orientation=180, center=(10, 10), width=0.5, layer=1),
        Port(name="p6", orientation=270, center=(10, 20), width=0.5, layer=1),
    ]
    assert not is_invalid_bundle_topology(p1, p2)

    p1 = [
        Port(name="p1", orientation=1e-11, center=(0, 0), width=0.5, layer=1),
        Port(name="p2", orientation=90, center=(0, 10), width=0.5, layer=1),
    ]
    p2 = [
        Port(name="p3", orientation=180, center=(10, 0), width=0.5, layer=1),
        Port(name="p4", orientation=180 + 1e-11, center=(10, 10), width=0.5, layer=1),
    ]
    assert not is_invalid_bundle_topology(p1, p2)
