"""This is a sample on how to define custom components."""
import shutil
from pathlib import Path

import gdsfactory as gf
from gdsfactory.add_grating_couplers import (
    add_grating_couplers_with_loopback_fiber_array,
)
from gdsfactory.component import Component
from gdsfactory.components.spiral_inner_io import spiral_inner_io
from gdsfactory.config import CONFIG, logger
from gdsfactory.sweep.write_sweeps import write_sweeps

add_te = gf.partial(
    gf.routing.add_fiber_array,
    grating_coupler=gf.components.grating_coupler_elliptical_te,
)


add_tm = gf.partial(
    gf.routing.add_fiber_array,
    grating_coupler=gf.components.grating_coupler_elliptical_tm,
)


@gf.cell
def coupler_te(gap: float, length: float) -> Component:
    """Directional coupler with TE grating couplers."""
    c = gf.components.coupler(gap=gap, length=length)
    cc = add_te(c)
    return cc


@gf.cell
def spiral_te(width: float = 0.5, length: float = 20e3) -> Component:
    """Waveguide Spiral with TE grating_coupler

    Args:
        width: um
        length: um
    """
    c = spiral_inner_io(width=width, length=length)
    c = gf.c.extend_ports(c)
    cc = add_grating_couplers_with_loopback_fiber_array(
        component=c,
        grating_coupler=gf.components.grating_coupler_elliptical_te,
        bend=gf.components.bend_circular,
    )
    return cc


@gf.cell
def spiral_tm(width: float = 0.5, length: float = 20e3) -> Component:
    """Waveguide Spiral with TM grating_coupler.

    Args:
        width: um
        lenght: um
    """
    c = spiral_inner_io(width=width, length=length, dx=10, dy=10, N=5)
    c = gf.c.extend_ports(c)
    cc = add_grating_couplers_with_loopback_fiber_array(
        component=c,
        grating_coupler=gf.components.grating_coupler_elliptical_tm,
        bend=gf.components.bend_circular,
    )
    return cc


component_factory = dict(
    spiral_te=spiral_te, spiral_tm=spiral_tm, coupler_te=coupler_te
)


def test_mask(precision: float = 2e-9) -> Path:
    from gdsfactory.autoplacer.yaml_placer import place_from_yaml

    workspace_folder = CONFIG["samples_path"] / "mask_custom"
    build_path = workspace_folder / "build"
    doe_root_path = build_path / "cache_doe"
    doe_metadata_path = build_path / "sweep"
    mask_path = build_path / "mask"
    does_yml = workspace_folder / "does.yml"

    shutil.rmtree(build_path, ignore_errors=True)
    mask_path.mkdir(parents=True, exist_ok=True)

    gdspath = mask_path / "sample_mask.gds"
    logpath = gdspath.with_suffix(".log")
    logger.add(sink=logpath)

    write_sweeps(
        str(does_yml),
        component_factory=component_factory,
        precision=precision,
        doe_root_path=doe_root_path,
        doe_metadata_path=doe_metadata_path,
    )

    top_level = place_from_yaml(does_yml, precision=precision, root_does=doe_root_path)
    top_level.write(str(gdspath))

    assert gdspath.exists()
    return gdspath


if __name__ == "__main__":
    # gdspath_mask = test_mask()
    # gf.show(gdspath_mask)
    # c = coupler_te(gap=0.3, length=2.0)
    # c = spiral_te(length=60e3)
    # c.show()

    # lengths = [18.24, 36.48, 54.72, 72.96, 91.2]
    # for length in lengths:
    #     c = coupler_te(gap=0.3, length=length)

    gds = test_mask()
    gf.show(gds)
