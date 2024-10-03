from __future__ import annotations

import gdsfactory as gf
from gdsfactory import cell
from gdsfactory.component import Component
from gdsfactory.typings import Callable, ComponentSpec, Float2, Iterable


@cell
def array(
    component: ComponentSpec = "pad",
    spacing: tuple[float, float] = (150.0, 150.0),
    columns: int = 6,
    rows: int = 1,
    add_ports: bool = True,
    size: Float2 | None = None,
    centered: bool = False,
    post_process: Iterable[Callable] | None = None,
) -> Component:
    """Returns an array of components.

    Args:
        component: to replicate.
        spacing: x, y spacing.
        columns: in x.
        rows: in y.
        add_ports: add ports from component into the array.
        size: Optional x, y size. Overrides columns and rows.
        centered: center the array around the origin.
        post_process: function to apply to the array after creation.

    Raises:
        ValueError: If columns > 1 and spacing[0] = 0.
        ValueError: If rows > 1 and spacing[1] = 0.

    .. code::

        2 rows x 4 columns
         ___        ___       ___          ___
        |   |      |   |     |   |        |   |
        |___|      |___|     |___|        |___|

         ___        ___       ___          ___
        |   |      |   |     |   |        |   |
        |___|      |___|     |___|        |___|
    """
    if size:
        columns = int(size[0] / spacing[0])
        rows = int(size[1] / spacing[1])

    if rows > 1 and spacing[1] == 0:
        raise ValueError(f"rows = {rows} > 1 require spacing[1] > 0")

    if columns > 1 and spacing[0] == 0:
        raise ValueError(f"columns = {columns} > 1 require spacing[0] > 0")

    c = Component()
    component = gf.get_component(component)
    ref = c.add_ref(component, columns=columns, rows=rows, spacing=spacing)
    old_center = ref.dcenter
    ref.dcenter = (0, 0) if centered else old_center

    if add_ports and component.ports:
        for ix in range(ref.na):
            for iy in range(ref.nb):
                for port in component.ports:
                    port = port.copy(ref.trans * gf.kdb.Trans(ix * ref.a + iy * ref.b))
                    name = f"{port.name}_{iy+1}_{ix+1}"
                    c.add_port(name, port=port)

    if post_process:
        for f in post_process:
            f(c)
    return c


if __name__ == "__main__":
    from functools import partial

    from gdsfactory.generic_tech import get_generic_pdk

    PDK = get_generic_pdk()
    PDK.activate()

    c = gf.components.array(
        partial(gf.components.straight, layer=(2, 0)),
        rows=3,
        columns=1,
        spacing=(0, 50),
        centered=False,
    )
    c.show()

    # c2 = array(rows=2, columns=2, spacing=(100, 100))
    # c2 = array(pad, rows=2, spacing=(200, 200), columns=1)
    # c3 = c2.copy()

    # c2 = array(pad, spacing=(200, 200), size=(700, 300), centered=False)

    # nports = len(c2.get_ports_list(orientation=0))
    # assert nports == 2, nports
    # c2.show( )
    # c2.show(show_subports=True)
    # c2.show()
