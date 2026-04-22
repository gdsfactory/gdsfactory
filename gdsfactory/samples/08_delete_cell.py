"""Delete a cell from the layout."""

if __name__ == "__main__":
    import gdsfactory as gf

    gf.gpdk.PDK.activate()

    c = gf.Component()
    ref1 = c << gf.components.text("1")
    ref2 = c << gf.components.text("2")

    print("Cells before:", len(list(gf.kcl.each_cell())))

    # Delete a specific cell from the layout
    gf.kcl.delete_cell(ref2.cell)

    print("Cells after:", len(list(gf.kcl.each_cell())))
    c.show()

    # Or clear the entire cache (all cells)
    # gf.clear_cache()
