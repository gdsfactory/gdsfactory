from __future__ import annotations

import os
import pickle

import gdsfactory as gf


def ports_are_equal(port1, port2) -> bool:
    return (
        port1.x == port2.x
        and port1.y == port2.y
        and port1.width == port2.width
        and port1.orientation == port2.orientation
        and port1.layer == port2.layer
    )


def components_are_equal(component1, component2) -> bool:
    # Compare the basic metadata
    if component1.name != component2.name or component1.settings != component2.settings:
        return False
    # print('basic metadata is equal')

    # Compare the ports
    if set(component1.ports.keys()) != set(component2.ports.keys()):
        return False
    # print('ports names are equal')

    for port_name in component1.ports:
        if not ports_are_equal(
            component1.ports[port_name], component2.ports[port_name]
        ):
            return False

    # print('ports are equal')
    # Compare the polygons
    polygons1 = component1.get_polygons(by_spec=True)
    polygons2 = component2.get_polygons(by_spec=True)
    if set(polygons1.keys()) != set(polygons2.keys()):
        return False
    # print('polygons keys are equal')
    return all(
        all((p1 == p2).all() for p1, p2 in zip(polygons1[spec], polygons2[spec]))
        for spec in polygons1.keys()
    )


def test_component_pickle() -> None:
    c1 = gf.components.straight()
    with open("test.pkl", "wb") as f:
        pickle.dump(c1, f)
    with open("test.pkl", "rb") as f:
        c2 = pickle.load(f)
    os.remove("test.pkl")

    assert components_are_equal(
        c1, c2
    ), "The components are not equal after pickling and unpickling."


if __name__ == "__main__":
    # test_component_pickle()
    c1 = gf.components.straight()
    with open("test.pkl", "wb") as f:
        pickle.dump(c1, f)
    with open("test.pkl", "rb") as f:
        c2 = pickle.load(f)
    os.remove("test.pkl")
    print(components_are_equal(c1, c2))
