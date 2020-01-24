""" read port markers from a GDS

"""

import csv
from pp import klive
import phidl.geometry as pg


def read_port_markers(gdspath, layer=69):
    """ loads a GDS and read port

    Args:
        gdspath:
        layer: GDS layer
    """
    D = pg.import_gds(gdspath)
    D = pg.extract(D, layers=[layer])
    for e in D.elements:
        print(e.x, e.y)


def csv2port(csvpath):
    """ loads and reads ports from a CSV file
    returns a dict
    """
    ports = {}
    with open(csvpath, "r") as csvfile:
        rows = csv.reader(csvfile, delimiter=",", quotechar="|")
        for row in rows:
            ports[row[0]] = row[1:]

    return ports


if __name__ == "__main__":
    import os
    from pp import CONFIG

    name = "mmi1x2_WM1"
    gdspath = os.path.join(CONFIG["lib"], name, name + ".gds")
    csvpath = os.path.join(CONFIG["lib"], name, name + ".ports")
    klive.show(gdspath)
    # read_port_markers(gdspath, layer=66)
    p = csv2port(csvpath)
    print(p)
