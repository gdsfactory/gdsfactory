from __future__ import annotations

__all__ = ["splitter_chain"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec


@gf.cell_with_module_name(tags=["containers"])
def splitter_chain(
    splitter: ComponentSpec = "mmi1x2",
    columns: int = 3,
    bend: ComponentSpec = "bend_s",
) -> Component:
    """Chain of splitters.

    Args:
        splitter: splitter to chain.
        columns: number of splitters to chain.
        bend: bend to connect splitters.

    ```text
                 __o5
              __|
           __|  |__o4
      o1 _|  |__o3
          |__o2
    ```

    ```text
           __o2
      o1 _|
          |__o3
    ```
    """
    return cf.splitter_chain(
        splitter=splitter,
        columns=columns,
        bend=bend,
    )
