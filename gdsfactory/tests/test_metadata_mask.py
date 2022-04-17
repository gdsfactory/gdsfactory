import toolz

import gdsfactory as gf


@gf.cell
def mask(size=(1000, 1000)):
    ring_te = toolz.compose(gf.routing.add_fiber_array, gf.components.ring_single)
    rings = gf.grid([ring_te(radius=r) for r in [10, 20, 50]])
    c = gf.Component()
    c << gf.components.die(size=size)
    c << rings
    return c


# def test_mask_metadata():
#     m = mask()
#     gdspath = m.write_gds_with_metadata(gdspath="test_mask_metadata.gds")
#     labels_path = gdspath.with_suffix(".csv")
#     gf.mask.write_labels(gdspath=gdspath, layer_label=(66, 0))
#     mask_metadata = gf.mask.read_metadata(gdspath=gdspath)
#     tm = gf.mask.merge_test_metadata(
#         mask_metadata=mask_metadata, labels_path=labels_path
#     )
#     assert len(tm.keys()) == 3
#     return m


if __name__ == "__main__":
    # m = test_mask_metadata()
    m = mask()
    m.show()
