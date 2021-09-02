import gdspy

from gdsfactory.types import PathType


def check_duplicate_cells(gdspath: PathType):
    """
    FIXME at gdspy level
    """
    gdsii_lib = gdspy.GdsLibrary()
    gdsii_lib.read_gds(gdspath)
    # component = gf.import_gds(gdspath)
    # cells = component.get_dependencies()
    # cell_names = [cell.name for cell in list(cells)]
    # cell_names_unique = set(cell_names)

    # if len(cell_names) != len(set(cell_names)):
    #     for cell_name in cell_names_unique:
    #         cell_names.remove(cell_name)

    #     cell_names_duplicated = "\n".join(set(cell_names))
    #     raise ValueError(
    #         f"Duplicated cell names in {component.name}:\n{cell_names_duplicated}"
    #     )


if __name__ == "__main__":
    # check_duplicate_cells("gds.gds")

    gdspath = "gds.gds"
    gdsii_lib = gdspy.GdsLibrary()
    gdsii_lib.read_gds(gdspath)
