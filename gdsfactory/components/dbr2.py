from typing import Optional, Tuple

import picwriter.components as pc

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.waveguide_template import strip
from gdsfactory.types import ComponentFactory


@gf.cell
def dbr2(
    length: float = 10.0,
    period: float = 0.85,
    dc: float = 0.5,
    w1: float = 0.4,
    w2: float = 1.0,
    taper_length: float = 20.0,
    fins: bool = False,
    fin_size: Tuple[float, float] = (0.2, 0.05),
    port: Tuple[int, int] = (0, 0),
    direction: str = "EAST",
    waveguide_template: ComponentFactory = strip,
    waveguide_template_dbr: Optional[ComponentFactory] = None,
    **kwargs
) -> Component:
    """Distributed Bragg Reflector Cell class.
    Tapers the input straight to a periodic straight structure with varying width
    (1-D photonic crystal).

    Args:
       length: Length of the DBR region.
       period: Period of the repeated unit.
       dc: Duty cycle of the repeated unit (must be a float between 0 and 1.0).
       w1: Width of the thin section of the straight.  w1 = 0 corresponds to disconnected periodic blocks.
       w2: Width of the wide section of the straight
       taper_length: Length of the taper between the input/output straight and the DBR region.
       fins (boolean): If `True`, adds fins to the input/output straights.
        In this case a different template for the component must be specified.
       fin_size ((x,y) Tuple): Specifies the x- and y-size of the `fins`. Defaults to 200 nm x 50 nm
       waveguide_template_dbr: If `fins` above is True, a WaveguideTemplate (dbr_wgt) must be specified.
        This defines the layertype / datatype of the DBR (which will be separate from the input/output straights)
       port (tuple): Cartesian coordinate of the input port.  Defaults to (0,0).
       direction (string): Direction that the component will point *towards*,
       can be of type `'NORTH'`, `'WEST'`, `'SOUTH'`, `'EAST'`, OR an angle (float, in radians)
       waveguide_template:  WaveguideTemplate object
       waveguide_template_dbr: Picwriter WaveguideTemplate object
       wg_width: 0.5
       wg_layer: gf.LAYER.WG[0]
       wg_datatype: gf.LAYER.WG[1]
       clad_layer: gf.LAYER.WGCLAD[0]
       clad_datatype: gf.LAYER.WGCLAD[1]
       bend_radius: 10
       cladding_offset: 3

    .. code::

                 period
        <-----><-------->
                _________
        _______|

          w1       w2       ...  n times
        _______
               |_________



    .. plot::
      :include-source:

      import gdsfactory as gf

      c = gf.components.dbr2(length=10, period=0.85, dc=0.5, w2=1, w1=0.4)
      c.plot()

    """

    waveguide_template_dbr = waveguide_template_dbr or waveguide_template(wg_width=w2)

    c = pc.DBR(
        wgt=gf.call_if_func(waveguide_template, wg_width=w2, **kwargs),
        length=length,
        period=period,
        dc=dc,
        w_phc=w1,
        taper_length=taper_length,
        fins=fins,
        fin_size=fin_size,
        dbr_wgt=waveguide_template_dbr,
        port=port,
        direction=direction,
    )

    return gf.read.picwriter(c)


if __name__ == "__main__":

    # c = dbr2(length=10, period=0.85, dc=0.5, w2=1, w1=0.4, taper_length=20, fins=True)
    c = dbr2()
    c.show()
