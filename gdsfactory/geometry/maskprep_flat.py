try:
    import kfactory as kf
    from kfactory import kdb
except ImportError as e:
    print(
        "You can install `pip install gdsfactory[kfactory]` for using maskprep. "
        "And make sure you use python >= 3.10"
    )
    raise e

from typing import Optional

from gdsfactory.typings import Dict, Layer, LayerSpecs, PathType, Tuple


def size(region: kdb.Region, offset: float):
    return region.dup().size(int(offset * 1e3))


def boolean_or(region1: kdb.Region, region2: kdb.Region):
    return region1.__or__(region2)


def boolean_not(region1: kdb.Region, region2: kdb.Region):
    return kdb.Region.__sub__(region1, region2)


def copy(region: kdb.Region):
    return region.dup()


class Region(kdb.Region):
    def __iadd__(self, offset):
        """Adds an offset to the layer."""
        return size(self, offset)

    def __isub__(self, offset):
        """Adds an offset to the layer."""
        return size(self, offset)

    def __add__(self, element):
        if isinstance(element, (float, int)):
            return size(self, element)

        elif isinstance(element, kdb.Region):
            return boolean_or(self, element)
        else:
            raise ValueError(f"Cannot add type {type(element)} to region")

    def __sub__(self, element):
        if isinstance(element, (float, int)):
            return size(self, -element)

        elif isinstance(element, kdb.Region):
            return boolean_not(self, element)

    def copy(self):
        return self.dup()


class RegionCollection:
    """RegionCollection of region per layer.

    Args:
        layermap: dict of layernames to layer numbers {'WG': (1, 0)}
        filepath: to read GDS from.
    """

    def __init__(
        self, layermap: Dict[str, Layer], filepath: Optional[PathType] = None
    ) -> None:
        lib = kf.kcell.KCLayout()
        lib.read(filename=str(filepath))
        c = lib[0]

        for layername, layer in layermap.items():
            region = Region()
            layer = lib.layer(layer[0], layer[1])
            region.insert(c.begin_shapes_rec(layer))
            region.merge()
            setattr(self, layername, region)

        self.layermap = layermap
        self.lib = lib

    def write(self, filename, cellname: str = "out") -> kf.KCell:
        """Write gds.

        Args:
            filepath: gdspath.
            cellname: for top cell.
        """
        c = kf.KCell(cellname, self.lib)

        for layername, layer in self.layermap.items():
            region = getattr(self, layername)
            try:
                c.shapes(self.lib.layer(layer[0], layer[1])).insert(region)
            except TypeError:
                raise ValueError(
                    f"Unexpected type for region {layername!r}: {type(region)}"
                )
        c.write(filename)
        return c

    def __delattr__(self, element) -> None:
        setattr(self, element, Region())

    def get_fill(
        self,
        region,
        size: Tuple[float, float],
        spacing: Tuple[float, float],
        fill_layers: LayerSpecs,
        fill_name: str = "fill",
        fill_cell_name: str = "fill_cell",
    ) -> None:
        """Generates rectangular fill on a set of layers in the region specified.

        Args:
            region: to fill, usually the result of prior boolean operations.
            size: (x,y) dimensions of the fill cell (um).
            spacing: (x,y) spacing of the fill cell (um).
            fill_layers: layers of the fill cell (can be multiple).
            fill_name: fill cell name.
            fill_cell_name: fill cell name.
        """

        fill_cell = kf.KCell(fill_cell_name)
        for layer in fill_layers:
            layer = kf.kcl.layer(*layer)
            fill_cell << kf.cells.waveguide.waveguide(
                width=size[0], length=size[1], layer=layer
            )

        fc_index = fill_cell.cell_index()  # fill cell index
        fc_box = fill_cell.bbox().enlarged(spacing[0] / 2 * 1e3, spacing[1] / 2 * 1e3)
        fill_margin = kf.kdb.Point(0, 0)

        fill = kf.KCell(fill_name)
        return fill.fill_region(
            region, fc_index, fc_box, None, region, fill_margin, None
        )


if __name__ == "__main__":
    import kfactory as kf

    import gdsfactory as gf
    import gdsfactory.geometry.maskprep_flat as dp
    from gdsfactory.generic_tech.layer_map import LAYER as l

    c = gf.Component()
    ring = c << gf.components.coupler_ring()
    floorplan = c << gf.components.bbox(ring.bbox, layer=l.FLOORPLAN)
    c.write_gds("src.gds")

    d = dp.RegionCollection(filepath="src.gds", layermap=dict(l))
    fill_cell = d.get_fill(
        d.FLOORPLAN - d.WG, size=(0.1, 0.1), spacing=(0.1, 0.1), fill_layers=(l.WG,)
    )
    fill_cell.write("fill.gds")
    gf.show("fill.gds")
