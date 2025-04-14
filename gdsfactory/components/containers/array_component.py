from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, PostProcesses, Size


@gf.cell_with_module_name
def array(
    component: ComponentSpec = "pad",
    columns: int = 6,
    rows: int = 1,
    column_pitch: float = 150,
    row_pitch: float = 150,
    add_ports: bool = True,
    size: Size | None = None,
    centered: bool = False,
    post_process: PostProcesses | None = None,
    auto_rename_ports: bool = False,
) -> Component:
    """Returns an array of components.

    Args:
        component: to replicate.
        columns: in x.
        rows: in y.
        column_pitch: pitch between columns.
        row_pitch: pitch between rows.
        auto_rename_ports: True to auto rename ports.
        add_ports: add ports from component into the array.
        size: Optional x, y size. Overrides columns and rows.
        centered: center the array around the origin.
        post_process: function to apply to the array after creation.

    Raises:
        ValueError: If columns > 1 and spacing[0] = 0.
        ValueError: If rows > 1 and spacing[1] = 0.

    .. code::

        2 rows x 4 columns

          column_pitch
          <---------->
         ___        ___       ___        ___
        |   |      |   |     |   |      |   |
        |___|      |___|     |___|      |___|

         ___        ___       ___        ___
        |   |      |   |     |   |      |   |
        |___|      |___|     |___|      |___|
    """
    if size:
        columns = int(size[0] / column_pitch)
        rows = int(size[1] / row_pitch)

    if rows > 1 and row_pitch == 0:
        raise ValueError(f"rows = {rows} > 1 require {row_pitch=} > 0")

    if columns > 1 and column_pitch == 0:
        raise ValueError(f"columns = {columns} > 1 require {column_pitch} > 0")

    c = Component()
    component = gf.get_component(component)
    ref = c.add_ref(
        component,
        columns=columns,
        rows=rows,
        column_pitch=column_pitch,
        row_pitch=row_pitch,
    )
    old_center = ref.center
    ref.center = (0, 0) if centered else old_center

    if add_ports and component.ports:
        for ix in range(ref.na):
            for iy in range(ref.nb):
                for port in component.ports:
                    port = port.copy(ref.trans * gf.kdb.Trans(ix * ref.a + iy * ref.b))
                    name = f"{port.name}_{iy + 1}_{ix + 1}"
                    c.add_port(name, port=port)

    if post_process:
        for f in post_process:
            f(c)
    if auto_rename_ports:
        c.auto_rename_ports()
    return c
