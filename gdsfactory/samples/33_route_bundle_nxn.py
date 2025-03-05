"""Routing bundle requires end ports to be on the same orientation but input can be any orientation."""

from __future__ import annotations

import gdsfactory as gf

if __name__ == "__main__":
    c = gf.Component()

    top = c << gf.components.nxn(north=8, south=0, east=0, west=0)
    bot = c << gf.components.nxn(north=2, south=2, east=2, west=2, xsize=10, ysize=10)

    top.dmovey(100)

    routes = gf.routing.route_bundle(
        c,
        ports1=bot.ports,
        ports2=top.ports,
        radius=5,
        sort_ports=True,
        cross_section="strip",
    )

    c.show()
