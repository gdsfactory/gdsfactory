"""This is a sample on how to define custom components."""
import shutil
from typing import Tuple

import numpy as np

import gdsfactory as gf
from gdsfactory.add_grating_couplers import (
    add_grating_couplers_with_loopback_fiber_array,
)
from gdsfactory.component import Component
from gdsfactory.config import CONFIG
from gdsfactory.mask.write_labels import write_labels

layer_label = (200, 0)

add_te = gf.partial(
    gf.routing.add_fiber_array,
    grating_coupler=gf.components.grating_coupler_elliptical_te,
    layer_label=layer_label,
)
add_tm = gf.partial(
    gf.routing.add_fiber_array,
    grating_coupler=gf.components.grating_coupler_elliptical_tm,
    bend_radius=20,
    layer_label=layer_label,
)


@gf.cell
def coupler_te(
    gap: float,
    length: int,
) -> Component:
    """Evanescent coupler with TE grating coupler."""
    c = gf.components.coupler(gap=gap, length=length)
    cc = add_te(c)
    return cc


@gf.cell
def spiral_te(width: float = 0.5, length: int = 2) -> Component:
    """Spiral with TE grating_coupler

    Args:
        width: waveguide width um
        lenght: cm
    """
    c = gf.c.spiral_inner_io(width=width, length=length)
    ce = gf.c.extend_ports(c)
    cc = add_grating_couplers_with_loopback_fiber_array(
        component=ce,
        grating_coupler=gf.components.grating_coupler_elliptical_te,
        bend=gf.components.bend_euler,
        layer_label=layer_label,
        component_name=c.name,
    )
    return cc


@gf.cell
def spiral_tm(width=0.5, length=20e3):
    """Spiral with TM grating_coupler

    Args:
        width: waveguide width um
        lenght: um
    """
    c = gf.c.spiral_inner_io(width=width, length=length, waveguide_spacing=10, N=5)
    ce = gf.c.extend_ports(c)
    cc = add_grating_couplers_with_loopback_fiber_array(
        component=ce,
        grating_coupler=gf.components.grating_coupler_elliptical_tm,
        bend=gf.components.bend_euler,
        layer_label=layer_label,
        component_name=c.name,
    )
    return cc


def test_mask(
    precision: float = 1e-9,
    labels_prefix: str = "opt",
    layer_label: Tuple[int, int] = layer_label,
) -> Component:
    """Returns mask."""
    workspace_folder = CONFIG["samples_path"] / "mask_pack"
    build_path = workspace_folder / "build"
    mask_path = build_path / "mask"

    shutil.rmtree(build_path, ignore_errors=True)
    mask_path.mkdir(parents=True, exist_ok=True)

    gdspath = mask_path / "sample_mask.gds"
    # markdown_path = gdspath.with_suffix(".md")
    # json_path = gdspath.with_suffix(".json")
    # test_metadata_path = gdspath.with_suffix(".tp.json")

    components = [spiral_te(length=length) for length in np.array([2, 4, 6]) * 1e4]
    components += [coupler_te(length=length, gap=0.2) for length in [10, 20, 30, 40]]
    c = gf.pack(components)
    m = c[0]
    m.name = "sample_mask"
    m.write_gds_with_metadata(gdspath)

    csvpath = write_labels(
        gdspath=gdspath, prefix=labels_prefix, label_layer=layer_label
    )
    assert gdspath.exists()
    assert csvpath.exists()
    return m


if __name__ == "__main__":
    m = test_mask()
    m.show()
