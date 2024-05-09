import gdsfactory as gf

if __name__ == "__main__":
    c = gf.components.straight_heater_doped_strip(width=1)
    c = gf.c.add_fiber_array_optical_south_electrical_north(c)

    lyrdb = c.connectivity_check(port_types=("optical", "electrical"))
    filepath = gf.config.home / "errors.lyrdb"
    lyrdb.save(filepath)
    gf.show(c, lyrdb=filepath)
