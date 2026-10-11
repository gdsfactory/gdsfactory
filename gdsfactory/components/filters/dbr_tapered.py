from __future__ import annotations

__all__ = ["dbr_tapered"]

import gdsfactory as gf
from gdsfactory import Component
from gdsfactory import component_functions as cf
from gdsfactory.typings import CrossSectionSpec, Size


@gf.cell_with_module_name(tags=["filters"])
def dbr_tapered(
    length: float = 10.0,
    period: float = 0.85,
    dc: float = 0.5,
    w1: float = 0.4,
    w2: float = 1.0,
    taper_length: float = 20.0,
    fins: bool = False,
    fin_size: Size = (0.2, 0.05),
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    """Distributed Bragg Reflector Cell class.

    Tapers the input straight to a
    periodic straight structure with varying width (1-D photonic crystal).

    Args:
       length: Length of the DBR region.
       period: Period of the repeated unit.
       dc: Duty cycle of the repeated unit (must be a float between 0 and 1.0).
       w1: thin section width. w1 = 0 corresponds to disconnected periodic blocks.
       w2: wide section width.
       taper_length: between the input/output straight and the DBR region.
       fins: If `True`, adds fins to the input/output straights.
       fin_size: Specifies the x- and y-size of the `fins`. Defaults to 200 nm x 50 nm
       cross_section: cross_section spec.

    ```text
                 period
        <-----><-------->
                _________
        _______|
    ```

    ```text
          w1       w2       ...  n times
        _______
               |_________
    ```
    """
    return cf.dbr_tapered(
        length=length,
        period=period,
        dc=dc,
        w1=w1,
        w2=w2,
        taper_length=taper_length,
        fins=fins,
        fin_size=fin_size,
        cross_section=cross_section,
    )
