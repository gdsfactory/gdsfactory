import gdsfactory as gf


@gf.cell
def sample_fill() -> gf.Component:
    """Sample fill example."""
    c = gf.Component()
    _ = c << gf.c.die_frame_phix_dc()
    fill = gf.c.rectangle(layer="M3")
    c.fill(
        fill_cell=fill,
        fill_layers=[("FLOORPLAN", -100)],
        exclude_layers=[("WG", 100), ("M3", 100)],
        x_space=1,
        y_space=1,
    )
    return c


@gf.cell
def sample_power_plane() -> gf.Component:
    """Sample fill example."""
    c = gf.Component()
    # _ = c << gf.c.die_frame_phix_dc()
    _ = c << gf.get_component("transceiver_mzi")
    gf.add_padding(c, layers=("FLOORPLAN",))

    fill = gf.c.rectangle_with_slits(size=(5, 5), layer="M3")
    c.fill(
        fill_cell=fill,
        fill_layers=[("FLOORPLAN", -100)],
        exclude_layers=[("SLAB150", 50), ("M3", 10)],
        x_space=0,
        y_space=0,
    )
    return c
