# AsiticSpiral class to interface with ASITIC for arbitrary polygons generated using PCells or GDS
# WIP


class AsiticSpiral:
    """Create an Asitic Spiral object from input dict entry.

    Need to decompose to Rectangles (WIP).

    Name : NAME --The unique name of the spiral.
    Length : LEN --The length of the wire.
    Width : WID --The width of the wire.
    Metal Layer : METAL --The metal layer of the wire segment.
    Origin of Spiral : XORG : YORG --The physical origin of the wire relative to the lower left hand corner of the chip.
    Note that you can specify all of the above options through the command line. You also can specify the following
    optional parameters:
    Orientation : ORIENT --The orientation of the wire relative to the x-axis. Angles are specified in units of degrees \
    and the counter-clockwise direction is considered positive.
    """

    def __init__(self, polygon, layer, datatype):
        """Not missing."""
        self.poly = polygon
        self.lay = layer
        self.datatype = datatype
        self.spiral_name = layer

    @property
    def polygon(self):
        return self.poly

    @property
    def layer(self, value=""):
        return self.lay

    @property
    def width(self):
        """Width has to be the same in a single polygon ?? Do we have tapering structures or notches?"""
        pass

    def write_to_asitic(self):
        """Write the asitic specific commands given the input polygons."""
        pass
