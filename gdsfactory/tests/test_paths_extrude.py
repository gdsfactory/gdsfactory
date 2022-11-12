import gdsfactory as gf
from gdsfactory.tech import LAYER


@gf.cell
def test_path_extrude_multiple_ports() -> gf.Component:
    s = gf.Section(
        width=2.0,
        offset=-4,
        layer=LAYER.HEATER,
        port_names=["e1", "e2"],
        port_types=("electrical", "electrical"),
    )
    X = gf.CrossSection(
        width=0.5, offset=0, layer=LAYER.SLAB90, port_names=["o1", "o2"], sections=[s]
    )
    P = gf.path.straight(npoints=100, length=10)
    c = gf.path.extrude(P, X)
    assert c.ports["e1"].port_type == "electrical"
    assert c.ports["e2"].port_type == "electrical"
    assert c.ports["o1"].port_type == "optical"
    assert c.ports["o2"].port_type == "optical"
    return c


if __name__ == "__main__":
    c = test_path_extrude_multiple_ports()
    c.show(show_ports=True)
