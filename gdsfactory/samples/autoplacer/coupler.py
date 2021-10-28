import gdsfactory as gf
from gdsfactory.components.coupler import coupler
from gdsfactory.routing.add_fiber_array import add_fiber_array


@gf.cell
def coupler2x2(gap=0.3, length=10.0):
    """Generate a coupler DOE."""
    c = coupler(gap=gap, length=length)
    return add_fiber_array(component=c)


if __name__ == "__main__":
    c = coupler2x2()
    c.show()
