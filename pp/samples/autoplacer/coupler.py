"""
Generate a coupler DOE
"""
import pp
from pp.routing.connect_component import add_io_optical
from pp.components.coupler import coupler


@pp.autoname
def CP2x2(gap=0.3, length=10.0):
    c = coupler(gap=gap, length=length)
    return add_io_optical(c)


if __name__ == "__main__":
    c = CP2x2()
    pp.show(c)
