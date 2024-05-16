import gdsfactory as gf
import gdsfactory.schematic as gt


def test_schematic_mzis():
    s = gt.Schematic()
    s.add_instance("mzi1", gt.Instance(component=gf.c.mzi(delta_length=10)))
    s.add_instance("mzi2", gt.Instance(component=gf.c.mzi(delta_length=100)))
    s.add_instance("mzi3", gt.Instance(component=gf.c.mzi(delta_length=200)))
    s.add_placement("mzi1", gt.Placement(x=000))
    s.add_placement("mzi2", gt.Placement(x=100, y=100))
    s.add_placement("mzi3", gt.Placement(x=200))
    s.add_net(gt.Net(ip1="mzi1,o2", ip2="mzi2,o2"))
    s.add_net(gt.Net(ip1="mzi2,o2", ip2="mzi3,o1"))
    assert s


def test_schematic_settings():
    n = 2**3
    splitter = gf.components.splitter_tree(noutputs=n)
    dbr_array = gf.components.array(
        component=gf.c.dbr, rows=n, columns=1, spacing=(0, 3), centered=True
    )
    s = gt.Schematic()
    s.add_instance("s", gt.Instance(component=splitter))
    s.add_placement("s", gt.Placement(x=0))
    s.add_instance("dbr", gt.Instance(component=dbr_array))
    s.add_placement("dbr", gt.Placement(x=100))

    for i in range(n):
        s.add_net(
            gt.Net(
                ip1=f"s,o2_2_{i+1}",
                ip2=f"dbr,o1_{i+1}_1",
                name="splitter_to_dbr",
                settings=dict(radius=5, enforce_port_ordering=False),
            )
        )
    assert len(s.netlist.routes["splitter_to_dbr"].links) == n


if __name__ == "__main__":
    n = 2**3
    splitter = gf.components.splitter_tree(noutputs=n)
    dbr_array = gf.components.array(
        component=gf.c.dbr, rows=n, columns=1, spacing=(0, 3), centered=True
    )
    s = gt.Schematic()
    s.add_instance("s", gt.Instance(component=splitter))
    s.add_placement("s", gt.Placement(x=0))
    s.add_instance("dbr", gt.Instance(component=dbr_array))
    s.add_placement("dbr", gt.Placement(x=100))

    for i in range(n):
        s.add_net(
            gt.Net(
                ip1=f"s,o2_2_{i+1}",
                ip2=f"dbr,o1_{i+1}_1",
                name="splitter_to_dbr",
                settings=dict(radius=5, enforce_port_ordering=False),
            )
        )

    print(s.netlist.routes)
    assert len(s.netlist.routes["splitter_to_dbr"].links) == n
