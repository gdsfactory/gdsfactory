from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.cross import cross
from gdsfactory.components.pad import pad_array
from gdsfactory.types import ComponentFactory, Optional


@cell
def add_fidutials(
    component: ComponentFactory = pad_array,
    gap: float = 50,
    left: Optional[ComponentFactory] = cross,
    right: Optional[ComponentFactory] = cross,
    top: Optional[ComponentFactory] = None,
    bottom: Optional[ComponentFactory] = None,
    **kwargs
) -> Component:
    """Return component with fidutials.

    Args:
        component: component to add to the new component.
        gap: from component to fidutial edge.
        left: optional left fidutial.
        right: optional right fidutial.
        top: optional top fidutial.
        bottom: optional bottom fidutial.

    """
    c = Component()
    component = component(**kwargs)
    r = c << component

    if left:
        x1 = c << left()
        x1.xmax = r.xmin - gap

    if right:
        x2 = c << right()
        x2.xmin = r.xmax + gap

    if top:
        y1 = c << top()
        y1.ymin = r.ymax + gap

    if bottom:
        y2 = c << bottom()
        y2.ymin = r.ymin - gap

    c.add_ports(r.ports)
    c.copy_child_info(component)
    return c


if __name__ == "__main__":
    c = add_fidutials(top=cross)
    c.show()
