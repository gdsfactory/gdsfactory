import gdsfactory as gf


def test_blackbox() -> None:
    """Blackbox keeps footprint and ports, hides everything else."""
    original = gf.components.mmi1x2()
    c = gf.components.blackbox(component="mmi1x2", layer=(2, 0))

    assert c.layers == [(2, 0)]
    assert c.dbbox() == original.dbbox()
    assert len(c.insts) == 0

    assert len(c.ports) == len(original.ports)
    for port, port_original in zip(c.ports, original.ports, strict=True):
        assert port.name == port_original.name
        assert port.center == port_original.center
        assert port.width == port_original.width
        assert port.orientation == port_original.orientation


def test_blackbox_does_not_modify_original() -> None:
    original = gf.components.mmi1x2()
    area_before = original.area((1, 0))
    original.to_blackbox(layer=(2, 0))
    assert original.area((1, 0)) == area_before
    assert (1, 0) in original.layers


def test_blackbox_settings_not_copied() -> None:
    """Settings of the original component must not leak into the blackbox."""
    c = gf.components.blackbox(component="mmi1x2")
    assert "length_mmi" not in c.settings


def test_to_blackbox_method() -> None:
    """Component.to_blackbox returns same result as the container."""
    original = gf.components.mmi1x2()
    c = original.to_blackbox(layer=(2, 0))

    assert c.layers == [(2, 0)]
    assert c.dbbox() == original.dbbox()
    assert len(c.insts) == 0
    assert [p.name for p in c.ports] == [p.name for p in original.ports]
