import gdsfactory as gf
import gdsfactory.typings as gt

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    s = gt.Schematic()
    s.add_instance("mzi1", gt.Instance(component=gf.c.mzi(delta_length=10)))
    s.add_instance("mzi2", gt.Instance(component=gf.c.mzi(delta_length=100)))
    s.add_instance("mzi3", gt.Instance(component=gf.c.mzi(delta_length=200)))
    s.add_placement("mzi1", gt.Placement(x=000))
    s.add_placement("mzi2", gt.Placement(x=100))
    s.add_placement("mzi3", gt.Placement(x=200))
    s.add_net(gt.Net(ip1="mzi1,o2", ip2="mzi2,o2"))
    s.add_net(gt.Net(ip1="mzi2,o2", ip2="mzi3,o1"))
    g = s.plot_netlist()
    plt.show()
