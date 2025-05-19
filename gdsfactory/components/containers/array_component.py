from __future__ import annotations

import numpy as np

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
    # --- Optimize: Only do int division if needed and inline variable assignment ---
    if size is not None:
        columns = int(size[0] // column_pitch)
        rows = int(size[1] // row_pitch)

    if rows > 1 and row_pitch == 0:
        raise ValueError(f"rows = {rows} > 1 require {row_pitch=} > 0")
    if columns > 1 and column_pitch == 0:
        raise ValueError(f"columns = {columns} > 1 require {column_pitch} > 0")

    c = Component()
    # --- Optimize: get_component once, reuse result ---
    base_component = gf.get_component(component)
    ref = c.add_ref(
        base_component,
        columns=columns,
        rows=rows,
        column_pitch=column_pitch,
        row_pitch=row_pitch,
    )
    # --- Optimize: Use ref.center only once ---
    if centered:
        ref.center = (0, 0)

    # --- Optimize port addition ---
    if add_ports and base_component.ports:
        # Precompute and cache values and lookups for speed
        ports = tuple(base_component.ports)
        trans = ref.trans
        a, b = ref.a, ref.b
        add_port = c.add_port
        name_cache = {}
        # Note: ref.na == columns, ref.nb == rows
        for ix in range(ref.na):
            for iy in range(ref.nb):
                tf = trans * gf.kdb.Trans(ix * a + iy * b)
                for port in ports:
                    # Avoid recomputing port name strings and port.copy if possible
                    pname = f"{port.name}_{iy + 1}_{ix + 1}"
                    if (pname, tf) in name_cache:
                        _port = name_cache[(pname, tf)]
                    else:
                        _port = port.copy(tf)
                        name_cache[(pname, tf)] = _port
                    add_port(pname, port=_port)

    if post_process:
        for f in post_process:
            f(c)
    if auto_rename_ports:
        c.auto_rename_ports()
    return c


def _get_rotated_basis(angle: float):
    """Fast helper, used by route_quad"""
    radians = np.deg2rad(angle)
    c, s = np.cos(radians), np.sin(radians)
    return np.array([c, s]), np.array([-s, c])
