"""FIXME.

How can we make sure that any port convention works for 3D FDTD.

see https://github.com/gdsfactory/gdsfactory/issues/233

This format could solve the issues

```

wavelength |port_in | port_out | mode_in | mode_out |   magnitude  |   phase
           |        |          |   0     |    0     |              |
           |        |          |         |          |              |
           |        |          |         |          |              |

```

you can write some convenience functions for each port naming convention
to query o1@0,o2@0 and s12a

df.query("port_in=='o1' & port_out=='o2' ")

"""

import gdsfactory as gf

if __name__ == "__main__":
    import gdsfactory.simulation.gtidy3d as gt

    c = gf.components.straight(length=2)
    c.unlock()
    c.auto_rename_ports_layer_orientation()
    # gt.write_sparameters(c, run=False)
    sp = gt.write_sparameters(c, run=True)
