import os

from pp.load_component import load_component


def hash_geometry(gdspath):
    """ the Geometrical hash includes only layers and polygons
    """
    assert os.path.isfile(gdspath), "!{} not found or not a valid GDS path".format(
        gdspath
    )
    name = gdspath.stem
    path = gdspath.parent
    c = load_component(name=name, dirpath=path)
    return c.hash_geometry()


def same_hash(gdspath1, gdspath2):
    """ returns True if hash is the same between 2 saved gds files
    ignores timestamps
    """
    return hash_geometry(gdspath1) == hash_geometry(gdspath2)
    # return gdspy.gdsii_hash(gdspath1) == gdspy.gdsii_hash(gdspath2)


if __name__ == "__main__":
    import pp

    gdspath = pp.write_component_type(
        "mmi1x2", width_mmi=4, overwrite=True, path_directory=pp.CONFIG["gdslib"]
    )
    print(hash_geometry(gdspath))

    gdspath = pp.write_component_type(
        "mmi1x2", width_mmi=5, overwrite=True, path_directory=pp.CONFIG["gdslib_test"]
    )
    print(hash_geometry(gdspath))
