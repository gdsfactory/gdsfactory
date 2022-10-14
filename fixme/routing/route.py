"""Manhattan routes fail when there is not enough space.

Now we raise a warning

ideally we also enable 180 deg routing.

"""

import gdsfactory as gf

if __name__ == "__main__":
    c = gf.components.coupler()
    gc = gf.components.grating_coupler_elliptical_te(width=1.0)
    cc = gf.routing.add_fiber_single(
        component=c,
        auto_widen=False,
        with_loopback=False,
        grating_coupler=gc,
        radius=40,
    )
    cc.show(show_ports=True)
