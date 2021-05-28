from pp.cell import cell
from pp.component import Component
from pp.components.straight import straight
from pp.port import auto_rename_ports


@cell
def array(
    component: Component = straight, n: int = 2, pitch: float = 20.0, axis: str = "y"
) -> Component:
    """Returns an array of components.

    Args:
        component: to replicate
        n: number of components
        pitch: float
        axis: X or y
    """
    c = Component()
    component = component() if callable(component) else component

    if axis not in ["x", "y"]:
        raise ValueError(f"Axis must be x or y, got {axis}")

    for i in range(n):
        ref = component.ref()
        if axis == "x":
            ref.x = i * pitch
        else:
            ref.y = i * pitch
        c.add(ref)
        for port in ref.get_ports_list():
            c.add_port(f"{port.name}_{i}", port=port)
    auto_rename_ports(c)
    return c


if __name__ == "__main__":

    # c1 = pp.c.pad()
    # c2 = array(component=c1, pitch=150, n=2)
    # print(c2.ports.keys())

    c2 = array()
    c2.show(show_ports=True)
