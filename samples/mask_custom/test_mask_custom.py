"""
This is a sample on how to define custom components.
You can make a repo out of this file, having one custom component per file
"""
import pathlib
import pp
from pp.config import load_config
from pp.config import CONFIG
from pp.autoplacer.yaml_placer import place_from_yaml
from pp.build import build_does
from pp.components.spiral_inner_io import spiral_inner_io_euler
from pp.add_termination import add_gratings_and_loop_back
from pp.routing.connect import connect_strip_way_points
from pp.add_padding import add_padding_to_grid
from pp.generate_does import generate_does

# from pp.placer import generate_does

def _route_filter(*args, **kwargs):
    return connect_strip_way_points(
        *args, taper_factory=None, start_straight=5.0, end_straight=5.0, **kwargs
    )


def add_te(component, **kwargs):
    c = pp.routing.add_io_optical(
        component,
        grating_coupler=pp.c.grating_coupler_elliptical_te,
        route_filter=_route_filter,
        **kwargs,
    )
    # c.test = 'passive_optical_te'
    c = add_padding_to_grid(c)
    return c


def add_tm(component, **kwargs):
    c = pp.routing.add_io_optical(
        component,
        grating_coupler=pp.c.grating_coupler_elliptical_tm,
        route_filter=_route_filter,
        bend_radius=20,
        **kwargs,
    )
    c = add_padding_to_grid(c)
    return c


@pp.autoname2
def coupler_te(gap, length, wg_width=0.5, nominal_wg_width=0.5, name=None):
    """ sample of component cutback """
    c = pp.c.coupler(wg_width=wg_width, gap=gap, length=length)
    cc = add_te(c, component_name=name)
    return cc


@pp.autoname2
def spiral_te(wg_width=0.5, length=2, name=None):
    """ sample of component cutback

    Args:
        wg_width: um
        lenght: mm
    """
    c = spiral_inner_io_euler(wg_width=wg_width, length=length)
    cc = add_gratings_and_loop_back(
        component=c,
        grating_coupler=pp.c.grating_coupler_elliptical_te,
        bend_factory=pp.c.bend_circular,
        component_name=name,
    )
    return cc


@pp.autoname2
def spiral_tm(wg_width=0.5, length=2, name=None):
    """ sample of component cutback """
    c = spiral_inner_io_euler(wg_width=wg_width, length=length, dx=10, dy=10, N=5)
    cc = add_gratings_and_loop_back(
        component=c,
        grating_coupler=pp.c.grating_coupler_elliptical_tm,
        bend_factory=pp.c.bend_circular,
        component_name=name,
    )
    return cc


component_type2factory = {}
component_type2factory["spiral_te"] = spiral_te
component_type2factory["spiral_tm"] = spiral_tm
component_type2factory["coupler_te"] = coupler_te


def test_mask_custom():
    # workspace_folder = pathlib.Path(__file__).parent
    workspace_folder = CONFIG["samples_path"] / "mask_custom"
    does_yml = workspace_folder / "does.yml"
    config_yml = workspace_folder / "config.yml"
    config = load_config(config_yml)
    gdspath = config["mask"]["gds"]

    # Map the component factory names in the YAML file to the component factory
    # generate_does(config)
    # build_does(config, component_type2factory=component_type2factory)
    generate_does(str(does_yml), component_type2factory=component_type2factory)

    top_level = place_from_yaml(does_yml)
    top_level.write(str(gdspath))
    assert gdspath.exists()
    return gdspath


if __name__ == "__main__":
    # from pprint import pprint
    # pprint(component_type2factory)
    c = test_mask_custom()
    pp.klive.show(c)
