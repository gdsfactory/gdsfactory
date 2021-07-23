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
    c = pp.components.mzi2x2(waveguide=waveguide)
    cc = pp.routing.add_fiber_single(
        component=c, auto_widen=False, waveguide=waveguide, with_loopback=False
    )
    cc.show()
