import gdsfactory
from gdsfactory.components import bend_circular
from gdsfactory.path import spiral_archimedean


@gdsfactory.cell
def double_spiral(
    inner_radius: float,
    separation: float,
    number_of_loops: float,
    npoints: int,
    cross_section: gdsfactory.types.CrossSectionSpec,
    bend_factory: gdsfactory.types.ComponentFactory = bend_circular,
):
    """
    Adds a double spiral

    Args:
        inner_radius: inner radius of the spiral
        separation: separation between the loops
        number_of_loops: number of loops per spiral
        npoints: points for the spiral
        cross_section: cross-section to extrude the structure with
        bend_factory: factory for the bends in the middle of the double spiral

    Returns:
        double spiral component
    """
    component = gdsfactory.Component()

    bend = bend_factory(radius=inner_radius / 2, angle=180, cross_section=cross_section)
    bend1 = component.add_ref(bend).mirror()
    bend2 = component.add_ref(bend)
    bend2.connect("o2", bend1.ports["o1"])

    path = spiral_archimedean(
        inner_radius=inner_radius,
        separation=separation,
        number_of_loops=number_of_loops,
        npoints=npoints,
    )
    path.start_angle = 0

    spiral = path.extrude(cross_section=cross_section)
    spiral1 = component.add_ref(spiral).connect("o1", bend1.ports["o2"])
    spiral2 = component.add_ref(spiral).connect("o1", bend2.ports["o1"])

    component.add_port("o1", port=spiral1.ports["o2"])
    component.add_port("o2", port=spiral2.ports["o2"])

    return component


if __name__ == "__main__":
    c = double_spiral(10, 2, 3, 1000, "nitride")
    print(c.ports)
    c.plot()
