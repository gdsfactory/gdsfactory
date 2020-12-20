from typing import Callable, Optional, Tuple

import picwriter.components as pc

import pp
from pp.component import Component
from pp.components.waveguide_template import wg_strip
from pp.picwriter2component import picwriter2component


@pp.cell
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
    waveguide_template: Callable = wg_strip,
    waveguide_template_dbr: Optional[Callable] = None,
    **kwargs
) -> Component:
    """ Distributed Bragg Reflector Cell class.  Tapers the input waveguide to a periodic waveguide structure with varying width (1-D photonic crystal).

    Args:
       wgt (WaveguideTemplate):  WaveguideTemplate object
       length (float): Length of the DBR region.
       period (float): Period of the repeated unit.
       dc (float): Duty cycle of the repeated unit (must be a float between 0 and 1.0).
       w1 (float): Width of the thin section of the waveguide.  w1 = 0 corresponds to disconnected periodic blocks.
       w2 (float): Width of the wide section of the waveguide
       taper_length (float): Length of the taper between the input/output waveguide and the DBR region.  Defaults to 20.0.
       fins (boolean): If `True`, adds fins to the input/output waveguides.  In this case a different template for the component must be specified.  This feature is useful when performing electron-beam lithography and using different beam currents for fine features (helps to reduce stitching errors).  Defaults to `False`
       fin_size ((x,y) Tuple): Specifies the x- and y-size of the `fins`.  Defaults to 200 nm x 50 nm
       dbr_wgt (WaveguideTemplate): If `fins` above is True, a WaveguideTemplate (dbr_wgt) must be specified.  This defines the layertype / datatype of the DBR (which will be separate from the input/output waveguides).  Defaults to `None`
       port (tuple): Cartesian coordinate of the input port.  Defaults to (0,0).
       direction (string): Direction that the component will point *towards*, can be of type `'NORTH'`, `'WEST'`, `'SOUTH'`, `'EAST'`, OR an angle (float, in radians)
       waveguide_template (WaveguideTemplate): Picwriter WaveguideTemplate object
       wg_width: 0.5
       wg_layer: pp.LAYER.WG[0]
       wg_datatype: pp.LAYER.WG[1]
       clad_layer: pp.LAYER.WGCLAD[0]
       clad_datatype: pp.LAYER.WGCLAD[1]
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

      import pp

      c = pp.c.dbr2(length=10, period=0.85, dc=0.5, w2=1, w1=0.4)
      pp.plotgds(c)

    """

    waveguide_template_dbr = waveguide_template_dbr or waveguide_template(wg_width=w2)

    c = pc.DBR(
        wgt=pp.call_if_func(waveguide_template, wg_width=w2, **kwargs),
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

    return picwriter2component(c)


if __name__ == "__main__":
    import pp

    c = dbr2(length=10, period=0.85, dc=0.5, w2=1, w1=0.4, taper_length=20, fins=True)
    pp.show(c)
