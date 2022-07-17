"""FIXME!

How can we make sure that any port convention works for 3D FDTD.
"""

import gdsfactory as gf

if __name__ == "__main__":
    import gdsfactory.simulation.gtidy3d as gt

    c = gf.components.straight(length=2)
    c.unlock()
    c.auto_rename_ports_layer_orientation()
    gt.write_sparameters(c, run=False)
