"""Port marker would be nice to understand the Shear angle."""
import gdsfactory as gf

if __name__ == "__main__":
    P = gf.path.straight(length=10)

    s = gf.Section(width=3, offset=0, layer=gf.LAYER.SLAB90, name="slab")
    X1 = gf.CrossSection(
        width=1,
        offset=0,
        layer=gf.LAYER.WG,
        name="core",
        port_names=("o1", "o2"),
        sections=[s],
    )
    s2 = gf.Section(width=2, offset=0, layer=gf.LAYER.SLAB90, name="slab")
    X2 = gf.CrossSection(
        width=0.5,
        offset=0,
        layer=gf.LAYER.WG,
        name="core",
        port_names=("o1", "o2"),
        sections=[s2],
    )
    t = gf.path.transition(X1, X2, width_type="linear")
    c = gf.path.extrude(P, t, shear_angle_start=10, shear_angle_end=45)
    c.show(show_ports=True)
