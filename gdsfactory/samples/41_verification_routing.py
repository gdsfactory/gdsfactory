import gdsfactory as gf

if __name__ == "__main__":
    from gdsfactory.generic_tech.cells import (
        add_fiber_array_optical_south_electrical_north,
    )

    c1 = gf.components.straight_heater_doped_strip(width=1)
    c2 = add_fiber_array_optical_south_electrical_north(
        component=c1,
        electrical_port_names=["top_e1", "bot_e3"],
    )
    c2.show()

    # lyrdb = c.connectivity_check(port_types=("optical", "electrical"))
    # filepath = gf.config.home / "errors.lyrdb"
    # lyrdb.save(filepath)
    # gf.show(c, lyrdb=filepath)
