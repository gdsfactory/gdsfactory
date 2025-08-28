import gdsfactory as gf


@gf.cell
def mzi_with_bend(radius: float = 10) -> gf.Component:
    c = gf.Component()
    bend = c.add_ref(gf.components.bend_euler(radius=radius))
    mzi = c.add_ref(gf.components.mzi())
    bend.connect(port="o1", other=mzi.ports["o2"])
    c.add_port(name="o1", port=mzi.ports["o1"])
    c.add_port(name="o2", port=bend.ports["o2"])
    return c
