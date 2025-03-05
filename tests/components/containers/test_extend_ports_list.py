from functools import partial

import gdsfactory as gf


def test_extend_ports_list() -> None:
    """Test extend_ports_list with a taper and straight extension."""
    c0 = gf.components.taper()
    e = gf.components.containers.extend_ports_list(
        component_spec=c0, extension="straight"
    )
    assert len(e.ports) == len(c0.ports) + len(gf.components.straight().ports)

    port_names = {p.name for p in e.ports}
    assert len(port_names) == len(e.ports)  # No duplicate names


def test_extend_ports_list_with_ignore() -> None:
    ignore_ports = ["o1"]
    m = gf.components.mmi1x2()
    t = partial(gf.components.taper, width2=0.1)
    e = gf.components.containers.extend_ports_list(
        component_spec=m, extension=t, ignore_ports=ignore_ports
    )

    port_names = {p.name for p in e.ports}
    assert set(ignore_ports) - port_names == set()


if __name__ == "__main__":
    test_extend_ports_list()
    test_extend_ports_list_with_ignore()
