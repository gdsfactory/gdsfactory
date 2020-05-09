import pathlib
import pp

from layers import LAYER

cwd = pathlib.Path(__file__).parent.absolute()
gds = cwd / "gds"


layer = LAYER.WG
port_width = 0.5


def position2orientation(p):
    """ we assume that ports with x<0 are inputs (orientation=180deg)
    and ports with x>0 are outputs
    """
    if p[0] < 0:
        return 180
    else:
        return 0


def import_gds(gdsname):
    """ import gds from SIEPIC PDK
    """
    c = pp.import_gds(gds / f"{gdsname}.gds")

    for label in c.get_labels():
        if label.text.startswith("pin"):
            port = pp.Port(
                name=label.text,
                midpoint=label.position,
                width=port_width,
                orientation=position2orientation(label.position),
                layer=layer,
            )
            c.add_port(port)

    return c


if __name__ == "__main__":
    gdsname = "ebeam_y_1550"
    c = import_gds(gdsname)
    pp.show(c)
