"""
Builds a library and compares the gdshash of the new built GDS with the reference ones
"""

import os
import shutil

import gdspy
import git
import pytest
from jsondiff import diff

import pp
from pp import CONFIG
from pp.components import component_factory, component_names

path_library = CONFIG["gds"]

component_names_test = component_names - set(
    [
        "bezier",
        "compass",
        "component_lattice",
        "coupler90",
        "grating_coupler_elliptical2",
        "grating_coupler_tree",
        "mzi",
        "spiral_circular",
        "waveguide_heater",
        "waveguide_pin",
        "waveguide_array",
        "wg_heater_connected",
        "test_via",
    ]
)


def pull_library(path_library=path_library):
    try:
        if os.path.isdir(path_library):
            print("git pull: {}".format(path_library))
            g = git.cmd.Git(path_library)
            g.pull()
    except Exception as e:
        error = f"error pulling {path_library} repo"
        raise ValueError(error, e)


def lock_component(
    component_type,
    component_factory=component_factory,
    path_library=path_library,
    flatten=True,
):
    """ locks a component from the factory into the GDS lib

    TODO: flattening makes it slower for big mask
    """
    try:
        c = component_factory[component_type](cache=False)
        if flatten:
            c.flatten()
        gdspath = path_library / (component_type + ".gds")
        pp.write_component(c, gdspath=gdspath)
        assert gdspath.exists()
        return c
    except Exception as e:
        error = f"error building {component_type}"
        raise ValueError(error, e)


def lock_components_with_changes(
    components=component_names_test,
    component_factory=component_factory,
    path_library=path_library,
):
    """  locks only components whose hash changed
    """
    for component_type in components:
        same_hash = compare_component_hash(
            component_type=component_type,
            component_factory=component_factory,
            path_library=path_library,
        )
        if not same_hash:
            print(f"locking {component_type} to {path_library}")
            lock_component(
                component_type=component_type,
                component_factory=component_factory,
                path_library=path_library,
            )


def print_components_with_changes(
    component_factory=component_factory, path_library=path_library
):
    """  locks only components whose hash changed
    """
    for component_type, _ in component_factory.items():
        same_hash = compare_component_hash(
            component_type,
            component_factory=component_factory,
            path_library=path_library,
        )
        if not same_hash:
            print(f"[X] {component_type} changed hash")


@pytest.mark.skip
@pytest.mark.parametrize("component_type", component_names_test)
@pytest.mark.noautofixt
def test_components(
    component_type, component_factory=component_factory, path_library=path_library,
):
    assert compare_component_hash(
        component_type=component_type,
        component_factory=component_factory,
        path_library=path_library,
    ), f"{component_type} changed from component locked in the library {path_library}"


def rebuild_library(component_factory=component_factory, path_library=path_library):
    """ saves all component_factory components to the gdslib library
    """
    for component_type, _ in component_factory.items():
        lock_component(
            component_type=component_type,
            component_factory=component_factory,
            path_library=path_library,
        )
        print(f"Write `{component_type}`")


def _copy_component(src, dest):
    for ext in [".gds", ".json", ".ports"]:
        shutil.copy(src.with_suffix(ext), dest.with_suffix(ext))

    for ext in [".ports"]:
        try:
            shutil.copy(src.with_suffix(ext), dest.with_suffix(ext))
        except BaseException:
            pass


def compare_component_hash(
    component_type,
    component_factory=component_factory,
    path_library=path_library,
    path_test=CONFIG["gdslib_test"],
):
    """ raises Exception if component has changed from the library
    writes component if it does not exist

    Args:
        component_type:
        path_library:
        path_test:
    """

    component_new = lock_component(
        component_type, component_factory=component_factory, path_library=path_test,
    )
    component_new.name += "_new"

    gdspath_new = path_test / (component_type + ".gds")
    gdspath_library = path_library / (component_type + ".gds")

    if not gdspath_library.exists():
        lock_component(
            component_type=component_type,
            component_factory=component_factory,
            path_library=path_library,
        )
        print(f"writing new component {component_type} into {path_library}")
        return True

    component_library = pp.load_component(name=component_type, dirpath=path_library)
    component_library.name += "_lib"
    gdshash_new = gdspy.gdsii_hash(gdspath_new)
    gdshash_library = gdspy.gdsii_hash(gdspath_library)

    # gdshash_new = component_library.hash_geometry()
    # gdshash_library = component_new.hash_geometry()

    same_hash = gdshash_new == gdshash_library
    if not same_hash:
        error_hash = f"`{component_library}` hash(GDS) {gdspath_new} differs from the library {gdspath_library}, showing both cells in Klayout \n library = {component_factory.keys()}"
        error_settings = f"different settings: {diff(component_library.get_settings(), component_new.get_settings())}"
        c = pp.Component(name=component_type)
        c << component_new
        c << component_library
        pp.show(c)
        print(error_hash + error_settings)

    return same_hash


if __name__ == "__main__":
    # for component in list(component_names_test):
    #     test_components(component)
    lock_components_with_changes()
    # lock_component("grating_coupler_tree")
    # compare_component_hash("grating_coupler_tree")
    # rebuild_library()
    # lock_component("waveguide")
    # compare_component_hash("waveguide")
    # lock_component("ring_double_bus")
    # compare_component_hash("ring_double_bus")
    # compare_component_hash("coupler90")
    # print_components_with_changes()
