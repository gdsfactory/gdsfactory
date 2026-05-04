import gdsfactory as gf

from gpdk import PDK

excluded = {"all_cells"}


@gf.cell
def all_cells() -> gf.Component:
    """Returns a component with all cells in the PDK."""
    cells = []
    for name, cell in PDK.cells.items():
        if name in excluded:
            continue
        try:
            c = cell()
            if c is not None:
                cells.append(c)
        except Exception as e:
            print(f"Error getting component {name}: {e}")

    c = gf.Component()
    _ = c << gf.pack(cells, spacing=20)[0]
    return c
