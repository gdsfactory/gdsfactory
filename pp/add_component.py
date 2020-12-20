""" functions to manage different GDS libraries
"""

import os
import shutil

import pp
from pp.config import CONFIG, conf
from pp.gdsdiff.gdsdiff import gdsdiff
from pp.write_component import write_component

GDS_PATH_TMP = CONFIG["gdslib_test"]


def get_library_path(tech=conf.tech.name):
    return CONFIG["gdslib"] / tech


def get_component_path(component, tech=conf.tech.name):
    return get_library_path(tech) / (component.name + ".gds")


def is_same_as_in_library(
    component, tech=conf.tech.name, gdspath_tmp=CONFIG["gdslib_test"],
):
    """
    component: a pp.Component
    tech: the tech where to find a reference component
    """

    # # Check if the component is here
    # if not component_is_in_library(component, tech):
    # return False

    # Library component path
    lib_component_dir = get_library_path(tech)

    lib_hash = None
    lib_component = pp.load_component(name=component.name, dirpath=lib_component_dir)

    lib_hash = lib_component.hash_geometry()
    new_component_hash = component.hash_geometry()
    return new_component_hash == lib_hash


def component_is_in_library(component, tech=conf.tech.name):
    lib_component_path = get_component_path(component, tech)
    return os.path.exists(lib_component_path)


def _copy_component(src, dest):
    for ext in [".gds", ".ports", ".json"]:
        shutil.copy(src.with_suffix(ext), dest.with_suffix(ext))


def add_component(
    component,
    tech,
    add_ports_to_all_cells=False,
    with_confirmation=True,
    force_replace=False,
    add_port_pins=True,
):
    """
    Arg:
        component <pp.Component>
        tech <str>

    Add a component to a given tech
    """

    lib_component_path = get_component_path(component, tech)
    # If it is a new library, create the path
    dirname = os.path.dirname(lib_component_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # Save component to make sure we have all the port and json files
    new_component_path = GDS_PATH_TMP / (component.name + ".gds")

    params = {
        "add_port_pins": add_port_pins,
        "add_ports_to_all_cells": add_ports_to_all_cells,
    }

    write_component(component, new_component_path, **params)

    # Reload component (to make sure ports info is added etc...)
    component = pp.load_component(component.name, GDS_PATH_TMP)
    write_component(component, new_component_path, **params)

    # If component is not in library, simply add it
    if not component_is_in_library(component, tech) or force_replace:
        _copy_component(new_component_path, lib_component_path)
        # write_component(component, lib_component_path,
        # add_ports_to_all_cells=add_ports_to_all_cells)
        return

    # If it is already there, compare the hash
    if is_same_as_in_library(component, tech):

        # We can safely overwrite the component since the geometry has not changed
        if not hasattr(component, "properties"):
            component.properties = {}
        _copy_component(new_component_path, lib_component_path)
        print(
            "{} cell geometry unchanged (cell/port names may have changed).".format(
                component.name
            )
        )

    elif with_confirmation:
        # Show the diff
        try:
            gds_diff = gdsdiff(lib_component_path, component)
            pp.show(gds_diff)

            a = input(
                "Do you want to replace {} with new version? [Y/N]".format(
                    component.name
                )
            )
            if a == "Y":
                _copy_component(new_component_path, lib_component_path)
                print("Replaced component {} by new one".format(component.name))
            else:
                print("Did not modify the component")
        except Exception as e:
            print("Cannot show the diff")
            print(e)
    else:
        _copy_component(new_component_path, lib_component_path)
        print("Replaced component {} by new one".format(component.name))
