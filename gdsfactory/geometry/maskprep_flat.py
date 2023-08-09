import warnings

from gplugins.klayout.dataprep.regions import (
    Region,
    RegionCollection,
    boolean_not,
    boolean_or,
    copy,
    size,
)

message = """
gdsfactory.simulation have been moved to gplugins

Make sure you have gplugins installed and use gplugins instead of gdsfactory.simulation

You can replace:
    import gdsfactory.geometry.maskprep_flat as dp -> import gplugins.klayout.dataprep.regions as dp

You can install gplugins with:
    pip install gplugins
"""

warnings.warn(message)


__all__ = [
    "boolean_not",
    "boolean_or",
    "copy",
    "size",
    "Region",
    "RegionCollection",
]

if __name__ == "__main__":
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
