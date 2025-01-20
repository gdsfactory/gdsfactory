import pytest

import gdsfactory as gf


def test_add_termination_default() -> None:
    c = gf.components.straight(length=50)
    cc = gf.components.add_termination(component=c)
    assert len(cc.ports) == 0


def test_add_termination_specific_ports() -> None:
    c = gf.components.straight(length=50)
    cc = gf.components.add_termination(component=c, port_names=("o2",))
    assert "o2" not in cc.ports
    assert "o1" in cc.ports


def test_add_termination_custom_terminator() -> None:
    c = gf.components.straight(length=50)
    custom_terminator = gf.components.taper(width2=0.2)
    cc = gf.components.add_termination(component=c, terminator=custom_terminator)
    assert len(cc.ports) == 0


if __name__ == "__main__":
    pytest.main([__file__])
