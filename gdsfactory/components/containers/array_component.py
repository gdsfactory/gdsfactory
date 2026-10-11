from __future__ import annotations

__all__ = ["array"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, PostProcesses, Size


@gf.cell_with_module_name(tags=["containers"])
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
    """Returns a Component containing a regular array of references.

    For an editable array, use `Component.add_ref()` and change the reference's
    `na` (columns) and `nb` (rows). See
    [Resizing an existing array](https://gdsfactory.github.io/gdsfactory/notebooks/01_references/#resizing-an-existing-array).

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

        2 rows x 4 columns

    ```text
          column_pitch
          <---------->
         ___        ___       ___        ___
        |   |      |   |     |   |      |   |
        |___|      |___|     |___|      |___|
    ```

    ```text
         ___        ___       ___        ___
        |   |      |   |     |   |      |   |
        |___|      |___|     |___|      |___|
    ```
    """
    return cf.array(
        component=component,
        columns=columns,
        rows=rows,
        column_pitch=column_pitch,
        row_pitch=row_pitch,
        add_ports=add_ports,
        size=size,
        centered=centered,
        post_process=post_process,
        auto_rename_ports=auto_rename_ports,
    )
