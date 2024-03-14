import gdsfactory as gf
import gdsfactory.typings as gt

if __name__ == "__main__":
    s = gt.Schematic()
    s.add_instance("mzi1", gt.Instance(component=gf.c.mzi(delta_length=10)))
    s.add_instance("mzi2", gt.Instance(component=gf.c.mzi(delta_length=100)))
    s.add_placement("mzi2", gt.Placement(x=300))
    s.add_net(gt.Net("mzi1", "o2", "mzi2", "o1"))
    s.plot_netlist()
