import pp

from waveguide import waveguide
from bend_circular import bend_circular
from y_splitter import y_splitter


@pp.autoname
def mzi(delta_length=100):
    c = pp.c.mzi(
        L1=delta_length,
        straight_factory=waveguide,
        bend90_factory=bend_circular,
        coupler_factory=y_splitter,
    )
    return c


if __name__ == "__main__":
    c = mzi(delta_length=100)
    pp.show(c)
