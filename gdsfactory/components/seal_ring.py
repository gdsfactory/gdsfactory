import gdsfactory as gf
from gdsfactory.components.contact import contact
from gdsfactory.components.rectangle import rectangle

big_square = gf.partial(rectangle, size=(1300, 2600))


@gf.cell
def seal_ring(
    component: gf.types.ComponentOrFactory = big_square,
    seal: gf.types.ComponentFactory = contact,
    width: float = 10,
    padding: float = 10.0,
    with_north: bool = True,
    with_south: bool = True,
    with_east: bool = True,
    with_west: bool = True,
) -> gf.Component:
    """Returns a continuous seal ring boundary at the chip/die
    seal rings are useful to prevents cracks from spreading
    you can connect it to ground

    Args:
        component: to add seal ring around
        seal: function for the seal
        width: of the seal
        padding: from component to seal
        with_north: includes seal
        with_south: includes seal
        with_east: includes seal
        with_west: includes seal

    """
    c = gf.Component()
    component = component() if callable(component) else component
    size = component.size
    sx, sy = size
    sx = gf.snap.snap_to_grid(sx, nm=2)
    sy = gf.snap.snap_to_grid(sy, nm=2)

    # north south
    size_north_south = (sx + 2 * padding + 2 * width, width)
    size_east_west = (width, sy + 2 * padding)

    if with_north:
        north = c << seal(size=size_north_south)
        north.ymin = component.ymax + padding
        north.x = component.x

    if with_south:
        south = c << seal(size=size_north_south)
        south.ymax = component.ymin - padding
        south.x = component.x

    if with_east:
        east = c << seal(size=size_east_west)
        east.xmin = component.xmax + padding
        east.y = component.y

    if with_west:
        west = c << seal(size=size_east_west)
        west.xmax = component.xmin - padding
        west.y = component.y

    return c


if __name__ == "__main__":
    c = big_square()
    c << seal_ring(with_south=False)
    c.show()
