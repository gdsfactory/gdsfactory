"""
implement basic cell cache

"""

import pathlib

import pydantic

from gdsfactory.component import Component
from gdsfactory.config import home_path, logger
from gdsfactory.import_gds import import_gds
from gdsfactory.name import clean_name, clean_value, get_name_short
from gdsfactory.types import ComponentFactory, PathType

cwd = pathlib.Path.cwd()


@pydantic.validate_arguments
def filecache(
    component_function: ComponentFactory,
    dirpath: PathType = home_path / "filepath",
    overwrite: bool = False,
    flatten: bool = True,
    with_metadata: bool = False,
) -> Component:
    """implements a basic file cache.
    Only builds component function not found in file cache.
    Flattens cells to ensure no naming conflicts
    """
    name = get_name_short(clean_name(clean_value(component_function)))
    gdspath = dirpath / f"{name}.gds"

    if gdspath.exists() and not overwrite:
        logger.info(f"loading filecache GDS {gdspath}")
        component = import_gds(gdspath)
    else:
        logger.info(f"writing filecache GDS {gdspath}")
        component = component_function()
        if flatten:
            component.flatten()

        if with_metadata:
            component.write_gds_with_metadata(gdspath=gdspath)
        else:
            component.write_gds(gdspath=gdspath)
    return component


if __name__ == "__main__":
    import gdsfactory as gf

    # c = filecache(gf.c.mzi)
    # c = gf.c.mzi()
    c = filecache(gf.c.straight_heater_metal_90_90)
    c.show()
