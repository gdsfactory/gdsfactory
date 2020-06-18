import picwriter.components as pc
from pp.ports import auto_rename_ports

from pp.components.waveguide_template import wg_strip
from pp.picwriter2component import picwriter2component
import pp


@pp.autoname
def spiral(
    width=500,
    length=10e3,
    spacing=None,
    parity=1,
    port=(0, 0),
    direction="NORTH",
    waveguide=wg_strip,
):
    """ Picwriter Spiral

    Args:
       width (float): width of the spiral (i.e. distance between input/output ports)
       length (float): desired length of the waveguide (um)
       spacing (float): distance between parallel waveguides
       parity (int): If 1 spiral on right side, if -1 spiral on left side (mirror flip)
       port (tuple): Cartesian coordinate of the input port
       direction (string): Direction that the component will point *towards*, can be of type `'NORTH'`, `'WEST'`, `'SOUTH'`, `'EAST'`, OR an angle (float, in radians)
       waveguide: Picwriter waveguide definition

    Members:
       * **portlist** (dict): Dictionary with the relevant port information

    Portlist format:
       * portlist['input'] = {'port': (x1,y1), 'direction': 'dir1'}
       * portlist['output'] = {'port': (x2, y2), 'direction': 'dir2'}

        Where in the above (x1,y1) are the first elements of the spiral trace, (x2, y2) are the last elements of the spiral trace, and 'dir1', 'dir2' are of type `'NORTH'`, `'WEST'`, `'SOUTH'`, `'EAST'`, *or* an angle in *radians*.
        'Direction' points *towards* the waveguide that will connect to it.
    """
    c = pc.Spiral(
        pp.call_if_func(waveguide),
        width=width,
        length=length,
        spacing=spacing,
        parity=parity,
        port=port,
        direction=direction,
    )
    # print(f'length = {length/1e4:.2f}cm')
    c = picwriter2component(c)
    c = auto_rename_ports(c)
    return c


if __name__ == "__main__":
    c = spiral(length=10e3, width=500, pins=True)
    pp.show(c)
