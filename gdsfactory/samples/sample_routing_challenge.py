import gdsfactory as gf
from gdsfactory.gpdk import PDK

PDK.activate()

if __name__ == "__main__":
    c = gf.Component("sample_array_ports")
    m = c.add_ref(
        gf.components.mzi_phase_shifter(),
        rows=3,
        row_pitch=100,
    )

    w = c.add_ref(
        gf.components.straight(length=10),
        rows=6,
        row_pitch=20,
        columns=6,
        column_pitch=20,
    )
    w.rotate(-90)
    w.x = m.x
    w.ymax = m.ymin - 100
    gf.routing.route_bundle(
        c,
        [m.ports["o1", 0, 0], m.ports["o1", 0, 1], m.ports["o1", 0, 2]],
        [w.ports["o1", 0, 0], w.ports["o1", 0, 1], w.ports["o1", 0, 2]],
        cross_section="strip",
    )

    lyrdb = c.connectivity_check()
    c.show(lyrdb=lyrdb)
