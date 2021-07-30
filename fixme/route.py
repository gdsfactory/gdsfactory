"""
Manhattan routes fail when there is not enough space

Now we raise a warning

ideally we also enable:

- sbend routing
- 180 deg routing

"""

import pp

if __name__ == "__main__":
    waveguide = "nitride"
    c = pp.components.coupler(waveguide=waveguide)
    gc = pp.components.grating_coupler_elliptical_te(
        wg_width=1.0,
    )
    cc = pp.routing.add_fiber_single(
        component=c,
        auto_widen=False,
        waveguide=waveguide,
        with_loopback=False,
        grating_coupler=gc,
    )
    cc.show()
