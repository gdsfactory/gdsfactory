from __future__ import annotations

import toolz
from omegaconf import OmegaConf

import gdsfactory as gf


@gf.cell
def mask(size=(1000, 1000)):
    ring_te = toolz.compose(gf.routing.add_fiber_array, gf.components.ring_single)
    rings = [ring_te(radius=r) for r in [10, 10, 20]]
    rings = gf.grid(
        rings, add_ports_suffix=True, decorator=gf.add_labels.add_labels_to_ports_x_y
    )
    c = gf.Component()
    c << gf.components.die(size=size)
    c << rings
    c.add_ports(rings.ports)
    return c


# def test_mask_metadata() -> gf.Component:
#     m = mask()
#     gdspath = m.write_gds_with_metadata(gdspath="test_mask_metadata.gds")
#     labels_path = gdspath.with_suffix(".csv")
#     gf.labels.write_labels.write_labels_klayout(gdspath=gdspath, layer_label=(66, 0))

#     yamlpath = gdspath.with_suffix(".yml")
#     mask_metadata = OmegaConf.load(yamlpath)
#     tm = gf.labels.merge_test_metadata(
#         mask_metadata=mask_metadata, labels_path=labels_path
#     )
#     assert len(tm.keys()) == 3
#     return m


if __name__ == "__main__":
    # m = test_mask_metadata()
    # m = mask(decorator=gf.add_labels.add_labels_to_ports_x_y)
    # m.show(show_ports=False)

    m = mask()
    gdspath = m.write_gds(gdspath="test_mask_metadata.gds", with_metadata=True)
    labels_path = gdspath.with_suffix(".csv")
    gf.labels.write_labels.write_labels_klayout(gdspath=gdspath, layer_label=(66, 0))

    yamlpath = gdspath.with_suffix(".yml")
    mask_metadata = OmegaConf.load(yamlpath)
    tm = gf.labels.merge_test_metadata(
        mask_metadata=mask_metadata, labels_path=labels_path
    )
    assert len(tm.keys()) == 3
